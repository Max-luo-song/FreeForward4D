/* FreeForward4D project page.
   Each scene video stacks three layers vertically: rendered RGB, depth, object Gaussians.
   A canvas draws the left part of the divider from layer 0 and the right part from the
   chosen layer, so both sides always come from the same decoded frame. */
(function () {
  "use strict";

  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var narrow = window.matchMedia("(max-width: 720px)");
  var LAYERS = 3;
  var CONTEXT_STEP = 5;         // context frames inside each window: every fifth frame
  var LAYER_NAMES = { 1: "Depth", 2: "Object Gaussians" };

  function createPlayer(root) {
    var stage = root.querySelector(".stage");
    var canvas = root.querySelector(".stage-canvas");
    var ctx = canvas.getContext("2d");
    var handle = root.querySelector(".handle");
    var rightLabel = root.querySelector("[data-right-label]");
    var playButton = root.querySelector(".play");
    var modeButtons = root.querySelectorAll(".mode");
    var road = root.querySelector(".road");
    var ticks = root.querySelector(".road-ticks");
    var head = root.querySelector(".road-head");

    var video = document.createElement("video");
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.setAttribute("playsinline", "");
    video.preload = "auto";

    var pressed = root.querySelector('.mode[aria-pressed="true"]');
    var state = {
      split: 0.5,
      layer: pressed ? Number(pressed.dataset.layer) : 1,
      userPaused: reducedMotion,
      dragging: false,
      visible: true
    };

    function sizeCanvas() {
      var ratio = Math.min(window.devicePixelRatio || 1, 2);
      var width = Math.round(stage.clientWidth * ratio);
      var height = Math.round(stage.clientHeight * ratio);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }

    function draw() {
      var vw = video.videoWidth;
      var vh = video.videoHeight;
      if (!vw || !vh) { return; }
      sizeCanvas();
      var cw = canvas.width;
      var ch = canvas.height;
      var lh = vh / LAYERS;
      // On phones only the front camera (middle third) is shown.
      var sx = narrow.matches ? vw / 3 : 0;
      var sw = narrow.matches ? vw / 3 : vw;
      var s = state.split;
      var leftSrc = sw * s;
      var leftDst = cw * s;
      if (leftSrc > 0.5) {
        ctx.drawImage(video, sx, 0, leftSrc, lh, 0, 0, leftDst, ch);
      }
      if (sw - leftSrc > 0.5) {
        ctx.drawImage(video, sx + leftSrc, state.layer * lh, sw - leftSrc, lh, leftDst, 0, cw - leftDst, ch);
      }
      if (video.duration) {
        head.style.left = (100 * video.currentTime / video.duration) + "%";
      }
    }

    // One scheduled callback at a time; play/load events only start the loop if idle.
    var scheduled = false;
    function frameLoop() {
      scheduled = false;
      draw();
      if (scheduled) { return; }
      if ("requestVideoFrameCallback" in video) {
        scheduled = true;
        video.requestVideoFrameCallback(frameLoop);
      } else if (!video.paused) {
        scheduled = true;
        window.requestAnimationFrame(frameLoop);
      }
    }
    function startLoop() { if (!scheduled) { frameLoop(); } }

    function setSplit(fraction) {
      state.split = Math.min(1, Math.max(0, fraction));
      var percent = Math.round(state.split * 100);
      handle.style.setProperty("--split", (state.split * 100) + "%");
      handle.setAttribute("aria-valuenow", String(percent));
      handle.setAttribute("aria-valuetext", percent + "% rendered RGB");
      draw();
    }

    function setLayer(layer) {
      state.layer = layer;
      modeButtons.forEach(function (button) {
        button.setAttribute("aria-pressed", String(Number(button.dataset.layer) === layer));
      });
      rightLabel.textContent = LAYER_NAMES[layer];
      root.dataset.accent = String(layer);
      draw();
    }

    function updatePlayButton() {
      var paused = video.paused;
      playButton.classList.toggle("is-paused", paused);
      playButton.setAttribute("aria-label", paused ? "Play" : "Pause");
    }

    function tryPlay() {
      if (state.userPaused || !state.visible) { return; }
      var promise = video.play();
      if (promise && promise.catch) { promise.catch(function () { updatePlayButton(); }); }
    }

    function buildTicks(windows, windowFrames) {
      ticks.textContent = "";
      var total = windows * windowFrames;
      for (var frame = 0; frame < total; frame += 1) {
        if (frame % windowFrames % CONTEXT_STEP === 0) {
          var tick = document.createElement("span");
          tick.className = "road-tick";
          tick.style.left = (100 * frame / total) + "%";
          ticks.appendChild(tick);
        }
      }
    }

    function load(scene, options) {
      root.classList.remove("is-ready");
      buildTicks(options.windows, options.windowFrames);
      video.src = scene.src;
      video.load();
    }

    video.addEventListener("loadeddata", function () {
      root.classList.add("is-ready");
      draw();
      tryPlay();
      startLoop();
    });
    video.addEventListener("play", function () { updatePlayButton(); startLoop(); });
    video.addEventListener("pause", updatePlayButton);
    video.addEventListener("seeked", draw);

    function pointerFraction(event) {
      var rect = stage.getBoundingClientRect();
      return (event.clientX - rect.left) / rect.width;
    }
    stage.addEventListener("pointerdown", function (event) {
      state.dragging = true;
      stage.setPointerCapture(event.pointerId);
      setSplit(pointerFraction(event));
    });
    stage.addEventListener("pointermove", function (event) {
      if (state.dragging) { setSplit(pointerFraction(event)); }
    });
    ["pointerup", "pointercancel"].forEach(function (name) {
      stage.addEventListener(name, function () { state.dragging = false; });
    });

    handle.addEventListener("keydown", function (event) {
      var step = event.shiftKey ? 0.2 : 0.05;
      var next = null;
      if (event.key === "ArrowLeft" || event.key === "ArrowDown") { next = state.split - step; }
      if (event.key === "ArrowRight" || event.key === "ArrowUp") { next = state.split + step; }
      if (event.key === "Home") { next = 0; }
      if (event.key === "End") { next = 1; }
      if (next !== null) { event.preventDefault(); setSplit(next); }
    });

    modeButtons.forEach(function (button) {
      button.addEventListener("click", function () { setLayer(Number(button.dataset.layer)); });
    });

    playButton.addEventListener("click", function () {
      if (video.paused) {
        state.userPaused = false;
        tryPlay();
      } else {
        state.userPaused = true;
        video.pause();
      }
    });

    road.addEventListener("click", function (event) {
      if (!video.duration) { return; }
      var rect = road.getBoundingClientRect();
      video.currentTime = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)) * video.duration;
    });

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          state.visible = entry.isIntersecting;
          if (state.visible) { tryPlay(); } else if (!video.paused) { video.pause(); }
        });
      }, { threshold: 0.1 }).observe(stage);
    }

    window.addEventListener("resize", draw);
    narrow.addEventListener("change", draw);
    if (document.fonts && document.fonts.ready) { document.fonts.ready.then(draw); }

    setSplit(0.5);
    setLayer(state.layer);
    updatePlayButton();
    return { load: load };
  }

  /* Clip wall: small autoplaying tiles that open in the comparison player. */
  function createTile(scene, onPick) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "tile";
    var video = document.createElement("video");
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.setAttribute("playsinline", "");
    video.preload = "none";
    video.poster = scene.tilePoster;
    video.src = scene.tile;
    video.width = 960;
    video.height = 426;
    var label = document.createElement("span");
    label.textContent = scene.title;
    button.appendChild(video);
    button.appendChild(label);
    button.addEventListener("click", function () { onPick(scene, button); });
    return { button: button, video: video };
  }

  function setupScenes() {
    var data = window.FF4D_SCENES;
    var root = document.querySelector("[data-player]");
    var wall = document.querySelector(".tile-wall");
    var caption = document.querySelector("[data-player-caption]");
    var modelButtons = document.querySelectorAll("[data-model]");
    if (!data || !root || !wall) { return; }
    var player = createPlayer(root);

    var observer = null;
    if ("IntersectionObserver" in window && !reducedMotion) {
      observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) { entry.target.pause(); return; }
          var promise = entry.target.play();
          if (promise && promise.catch) { promise.catch(function () {}); }
        });
      }, { threshold: 0.25 });
    }

    function selectModel(key) {
      var model = data[key];
      var options = { windows: model.windows, windowFrames: model.windowFrames };
      modelButtons.forEach(function (button) {
        button.setAttribute("aria-pressed", String(button.dataset.model === key));
      });
      caption.textContent = model.caption;
      if (observer) { observer.disconnect(); }
      wall.textContent = "";

      function pick(scene, pressed) {
        wall.querySelectorAll(".tile").forEach(function (tile) {
          tile.setAttribute("aria-pressed", String(tile === pressed));
        });
        player.load(scene, options);
      }

      model.scenes.forEach(function (scene, index) {
        var tile = createTile(scene, function (chosen, button) {
          pick(chosen, button);
          root.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "nearest" });
        });
        wall.appendChild(tile.button);
        if (observer) { observer.observe(tile.video); }
        if (index === 0) { pick(scene, tile.button); }
      });
    }

    modelButtons.forEach(function (button) {
      button.addEventListener("click", function () { selectModel(button.dataset.model); });
    });
    selectModel("s2");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupScenes);
  } else {
    setupScenes();
  }
})();
