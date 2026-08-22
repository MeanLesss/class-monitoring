/*
 * Minimal Streamlit component bridge - implements the iframe postMessage
 * protocol directly (no external CDN script needed).
 */
(function () {
  function sendMsg(type, payload) {
    var msg = { isStreamlitMessage: true, type: type };
    if (payload) {
      for (var k in payload) msg[k] = payload[k];
    }
    window.parent.postMessage(msg, "*");
  }

  function setComponentReady() {
    sendMsg("streamlit:componentReady", { apiVersion: 1 });
  }

  function setComponentValue(value, dataType) {
    sendMsg("streamlit:setComponentValue", {
      value: value,
      dataType: dataType || "json",
    });
  }

  function setFrameHeight(height) {
    sendMsg("streamlit:setFrameHeight", { height: height });
  }

  function onRenderEvent(handler) {
    window.addEventListener("message", function (event) {
      if (event.data && event.data.type === "streamlit:render") {
        handler({
          detail: {
            args: event.data.args,
            disabled: event.data.disabled,
            theme: event.data.theme,
          },
        });
      }
    });
  }

  window.StreamlitComponentLib = {
    setComponentReady: setComponentReady,
    setComponentValue: setComponentValue,
    setFrameHeight: setFrameHeight,
    onRenderEvent: onRenderEvent,
  };
})();
