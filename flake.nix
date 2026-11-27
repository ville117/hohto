{
  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url  = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rust-toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        runtimeLibs = with pkgs; [
          wayland
          libxkbcommon
          vulkan-loader
          vulkan-validation-layers
          udev
          alsa-lib
          libx11
          libxcursor 
          libxi
          libxrandr
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            rust-toolchain
            pkgs.pkg-config
            pkgs.shader-slang
          ];
          buildInputs = runtimeLibs;
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
          RUST_SRC_PATH = "${rust-toolchain}/lib/rustlib/src/rust/library";
          VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        };
      }
    );
}
