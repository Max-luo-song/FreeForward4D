/* Scene videos: each file stacks rendered RGB, depth, and object Gaussians (top to bottom).
   `windows` is the number of consecutive reconstructions in a clip, `windowFrames` the recorded
   frames in each. Tiles are smaller clips of the RGB and object layers only. */
(function () {
  "use strict";

  function scenes(model, dir, list) {
    return list.map(function (item) {
      return {
        title: item[1],
        src: "assets/" + dir + "/" + item[0] + ".mp4",
        tile: "assets/tiles/" + model + "-" + item[0] + ".mp4",
        tilePoster: "assets/tiles/" + model + "-" + item[0] + ".jpg"
      };
    });
  }

  window.FF4D_SCENES = {
    s2: {
      windows: 6,
      windowFrames: 16,
      caption: "Drag the divider to compare. Each clip chains six 2 s windows, each reconstructed in one forward pass from 4 context frames per camera, marked on the timeline. Every frame shown is rendered, including frames halfway between camera timestamps. Gaussians of tracked objects are colored by object, following the grouping derived from 2D tracks.",
      scenes: scenes("s2", "scenes", [
        ["hero", "Palm avenue"],
        ["waterfront", "Waterfront"],
        ["avenue", "Wide avenue"],
        ["suburban", "Suburban traffic"],
        ["hillside", "Hillside street"],
        ["residential", "Residential street"],
        ["van", "Following a van"],
        ["rowhouses", "Row houses"]
      ])
    },
    s8: {
      windows: 1,
      windowFrames: 80,
      caption: "Drag the divider to compare. Each 8 s clip is reconstructed in one forward pass from 16 context frames per camera, marked on the timeline. Every frame shown is rendered, including frames halfway between camera timestamps. Gaussians of moving objects (predicted speed above 1 m/s) are colored by object, following the grouping derived from 2D tracks.",
      scenes: scenes("s8", "scenes8s", [
        ["hero", "Palm avenue"],
        ["freeway", "Dusk freeway"],
        ["avenue", "Wide avenue"],
        ["suburban", "Suburban traffic"],
        ["hillside", "Hillside street"],
        ["residential", "Residential street"],
        ["van", "Following a van"],
        ["rowhouses", "Row houses"]
      ])
    }
  };
})();
