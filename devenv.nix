{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/basics/
  env.GREET = "tiktok-engaging-project";

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.zlib
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    directory = "./";
    version = "3.12";
    venv = {
      enable = true;
    };
    uv = {
      enable = true;
      sync.enable = true;
    };
  };




# === === === === === === ===

  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    pkgs.zlib
  ];

# === === === === === === ===



  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo ==== $GREET ====
  '';

  # https://devenv.sh/basics/
  enterShell = ''
    hello         # Run scripts directly
    git --version # Use packages
  '';
}
