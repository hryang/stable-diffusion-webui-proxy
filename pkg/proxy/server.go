package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"

	"github.com/gorilla/websocket"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

type ReverseProxy struct {
	Target url.URL
	Proxy  *httputil.ReverseProxy
}

type Server struct {
	Echo        *echo.Echo // the echo server for reverse proxy
	UIServer    *ReverseProxy
	InferServer *ReverseProxy
}

// NewServer create the reverse proxy for the downstream inference server and ui server.
// The inference server is a remote GPU server, while the ui server is local.
// REQUIRES: the local server has been started. The remote inference server is ready.
func NewServer(inferTargetStr string, uiTargetStr string) *Server {
	s := &Server{
		Echo: echo.New(),
	}

	// s.Echo.Debug = true
	s.Echo.Use(middleware.Logger())
	s.Echo.Use(middleware.Recover())

	target, err := url.Parse(inferTargetStr)
	if err != nil {
		panic(fmt.Errorf("parse target %s failed: %v", inferTargetStr, err))
	}
	s.InferServer = &ReverseProxy{
		Target: *target,
		Proxy:  httputil.NewSingleHostReverseProxy(target),
	}
	s.Echo.Logger.Infof("create reverse proxy for inference server: %s", inferTargetStr)

	uiServerTarget, _ := url.Parse(uiTargetStr)
	s.UIServer = &ReverseProxy{
		Target: *uiServerTarget,
		Proxy:  httputil.NewSingleHostReverseProxy(uiServerTarget),
	}
	s.Echo.Logger.Infof("create reverse proxy for UI server: %s", uiTargetStr)

	s.Echo.GET("/queue/join", s.queueJoinHandler)

	// Task progress requests are handled by inference server.
	s.Echo.POST("/internal/progress", func(c echo.Context) error {
		req := c.Request()
		req.Host = s.InferServer.Target.Host
		req.URL.Host = s.InferServer.Target.Host
		req.URL.Scheme = s.InferServer.Target.Scheme
		s.InferServer.Proxy.ServeHTTP(c.Response(), c.Request())
		return nil
	})

	// All other cases are handled by ui server.
	s.Echo.Any("/*", func(c echo.Context) error {
		req := c.Request()
		req.Host = s.UIServer.Target.Host
		req.URL.Host = s.UIServer.Target.Host
		req.URL.Scheme = s.UIServer.Target.Scheme
		s.UIServer.Proxy.ServeHTTP(c.Response(), c.Request())
		return nil
	})

	return s
}

func (s *Server) Start(address string) error {
	return s.Echo.Start(address)
}

func (s *Server) queueJoinHandler(c echo.Context) error {
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
	}

	// Upgrade the client HTTP to WebSocket connection.
	clientConn, err := upgrader.Upgrade(c.Response().Writer, c.Request(), nil)
	if err != nil {
		return err
	}
	defer clientConn.Close()

	// Create the downstream UI server WebSocket connection.
	uiDialer := websocket.DefaultDialer
	downstream, _ := url.JoinPath("ws://", s.UIServer.Target.Host, "queue/join")
	uiServerConn, _, err := uiDialer.Dial(downstream, nil)
	if err != nil {
		return err
	}
	defer uiServerConn.Close()

	// Create the downstream inference server WebSocket connection.
	inferDialer := websocket.DefaultDialer
	downstream, _ = url.JoinPath("ws://", s.InferServer.Target.Host, "queue/join")
	inferServerConn, _, err := inferDialer.Dial(downstream, nil)
	if err != nil {
		return err
	}
	defer inferServerConn.Close()

	var wg sync.WaitGroup

	// Create goroutine to handle client-to-server request.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			// Read message.
			messageType, message, err := clientConn.ReadMessage()
			if _, ok := err.(*websocket.CloseError); ok {
				s.Echo.Logger.Infof("close the websocket connection.")
				return
			}
			if err != nil {
				s.Echo.Logger.Errorf("read from websocket client error: %v", err)
				return
			}
			s.Echo.Logger.Debugf("websocket send message: %s", string(message))

			// Parse task id from message.
			var m map[string]interface{}
			if err := json.Unmarshal(message, &m); err != nil {
				s.Echo.Logger.Errorf("unmarshal json error: %v", err)
				return
			}
			var taskId string
			if data, ok := m["data"]; ok {
				if l, ok := data.([]interface{}); ok {
					if len(l) > 0 {
						taskId, _ = l[0].(string)
					}
				}
			}
			s.Echo.Logger.Infof("websocket message: %v", string(message))

			// Forward the task-launching request to the inference server.
			// The task launching message contains the task id, which is the first element of the "data" array.
			// For example, following two messages, the first one is the task launching message, the second one is not.
			// {"fn_index": 89, "data": ["task(yx99r25qdxzgrue)", "city, cute boy", ...], ...}
			// {"fn_index": 94, "data": ["city, cute boy, ..."], ...}
			if strings.HasPrefix(taskId, "task") {
				err = inferServerConn.WriteMessage(messageType, message)
			} else {
				err = uiServerConn.WriteMessage(messageType, message)
			}

			if err != nil {
				s.Echo.Logger.Errorf("write to websocket server error: %v", err)
				return
			}
		}
	}()

	// Create goroutine to handle ui-server-to-client responses.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			messageType, message, err := uiServerConn.ReadMessage()
			if _, ok := err.(*websocket.CloseError); ok {
				s.Echo.Logger.Infof("close the ui server websocket connection")
				return
			}
			if err != nil {
				s.Echo.Logger.Errorf("read from websocket server error: %v", err)
				return
			}
			s.Echo.Logger.Debugf("ui server websocket receive response: %s", string(message))
			err = clientConn.WriteMessage(messageType, message)
			if err != nil {
				s.Echo.Logger.Errorf("write to websocket client error: %v", err)
				return
			}
		}
	}()

	// Create goroutine to handle inference-server-to-client responses.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			messageType, message, err := inferServerConn.ReadMessage()
			if _, ok := err.(*websocket.CloseError); ok {
				s.Echo.Logger.Info("close the inference server websocket connection")
				return
			}
			if err != nil {
				s.Echo.Logger.Errorf("read from websocket server error: %v", err)
				return
			}
			s.Echo.Logger.Debugf("inference server websocket receive resposne: %s", string(message))
			err = clientConn.WriteMessage(messageType, message)
			if err != nil {
				s.Echo.Logger.Errorf("write to websocket client error: %v", err)
				return
			}
		}
	}()

	// Wait for all goroutines finishing.
	// The echo framework handles the requests in seperated goroutines. So blocking-wait is OK.
	wg.Wait()

	return nil
}
