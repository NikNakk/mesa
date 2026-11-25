#!/bin/bash
set -euo pipefail
: "${COMPONENT:?}" "${PLATFORM:?}" "${VERSION:?}" "${PREFIX:?}"
root=$PWD
work="$root/.threshold-build"
mkdir -p "$work" "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/licenses"
for license in LICENSE LICENSE.* COPYING COPYING.* NOTICE NOTICE.*; do
  if [[ -f "$license" ]]; then cp "$license" "$PREFIX/licenses/"; fi
done
if [[ "$PLATFORM" == macos-arm64 ]]; then expected_arch=arm64; else expected_arch=x86_64; fi
[[ "$(uname -m)" == "$expected_arch" ]] || { echo "Wrong runner architecture" >&2; exit 1; }
export MACOSX_DEPLOYMENT_TARGET=14.0
if [[ "$(uname -s)" == Darwin ]]; then
  brew install cmake ninja pkgconf
elif [[ "$COMPONENT" == xrizer && "$PLATFORM" == windows-x86_64 ]]; then
  sudo apt-get update
  sudo apt-get install --no-install-recommends -y cmake ninja-build pkg-config mingw-w64 glslc clang libclang-dev
else
  echo "Unsupported build host" >&2; exit 2
fi
clone_input() {
  local url=$1 ref=$2 destination=$3
  git clone --depth 1 --branch "$ref" "$url" "$destination"
  printf '%s %s\n' "$url" "$(git -C "$destination" rev-parse HEAD)" >> "$PREFIX/build-inputs.txt"
}
build_threshold_foveation() {
  local revision=58c1aa3099fc83814c79440c6ee5036c7b0ef206
  local url=https://github.com/NikNakk/threshold-foveation.git
  local source=${THRESHOLD_FOVEATION_SOURCE:-$url}
  local checkout="$work/threshold-foveation-source"
  local install="$work/threshold-foveation"
  git init -q "$checkout"
  git -C "$checkout" fetch --depth 1 "$source" "$revision"
  git -C "$checkout" checkout --detach FETCH_HEAD
  printf '%s %s\n' "$url" "$(git -C "$checkout" rev-parse HEAD)" >> "$PREFIX/build-inputs.txt"
  cmake -S "$checkout" -B "$work/threshold-foveation-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX="$install"
  cmake --build "$work/threshold-foveation-build" --parallel 3
  cmake --install "$work/threshold-foveation-build"
  cp -P "$install/lib/"libthreshold_foveation*.dylib "$PREFIX/lib/"
  mkdir -p "$PREFIX/licenses/threshold-foveation"
  cp "$install/share/licenses/threshold-foveation/"* "$PREFIX/licenses/threshold-foveation/"
}
case "$COMPONENT" in
monado)
  brew install eigen glslang jpeg-turbo libusb molten-vk vulkan-headers vulkan-loader
  build_tag=$(git describe --tags --always --exclude 'threshold-*')
  opts=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_RPATH='@loader_path/../lib'
    -DBUILD_TESTING=OFF -DGIT_DESC="$build_tag" -DXRT_FEATURE_SERVICE=ON
    -DXRT_HAVE_OPENGLES=OFF -DXRT_HAVE_SDL2=OFF -DXRT_MODULE_MONADO_GUI=OFF
    -DXRT_FEATURE_WINDOW_PEEK=OFF -DXRT_FEATURE_DEBUG_GUI=OFF -DXRT_FEATURE_CLIENT_DEBUG_GUI=OFF
    -DXRT_BUILD_DRIVER_PSSENSE=OFF -DXRT_FEATURE_OPENXR_INTERACTION_KHR_GENERIC=ON)
  if [[ "$PLATFORM" == macos-x86_64 ]]; then
    opts+=(-DXRT_BUILD_DRIVER_PSVR2=OFF -DXRT_HAVE_LIBUSB=OFF -DXRT_HAVE_HIDAPI=OFF
      -DXRT_BUILD_DRIVER_HANDTRACKING=OFF -DXRT_FEATURE_OPENVR=OFF -DXRT_HAVE_OPENCV=OFF)
  else
    opts+=(-DXRT_BUILD_DRIVER_PSVR2=ON)
  fi
  cmake -S . -B "$work/build" -G Ninja "${opts[@]}"
  if [[ "$PLATFORM" == macos-x86_64 ]]; then
    cmake --build "$work/build" --target openxr_monado --parallel 3
    cp "$work/build/src/xrt/targets/openxr/libopenxr_monado.dylib" "$PREFIX/lib/"
    mkdir -p "$PREFIX/share/openxr/1"
    printf '%s\n' '{"file_format_version":"1.0.0","runtime":{"name":"Monado","library_path":"../../../lib/libopenxr_monado.dylib"}}' > "$PREFIX/share/openxr/1/openxr_monado.json"
  else
    cmake --build "$work/build" --parallel 3
    cmake --install "$work/build"
    test -x "$PREFIX/bin/monado-service"
    test -x "$PREFIX/bin/monado-service-xpc-control"
  fi
  test -f "$PREFIX/lib/libopenxr_monado.dylib"
  printf '%s' "$build_tag" > "$work/build-tag"
  ;;
xrizer)
  rustup update stable --no-self-update
  rustup override set stable
  if [[ "$(uname -s)" == Darwin ]]; then
    brew install shaderc vulkan-headers vulkan-loader molten-vk
  fi
  export XRIZER_VERSION="$VERSION"
  if [[ "$PLATFORM" == windows-x86_64 ]]; then
    if [[ "$(uname -s)" == Darwin ]]; then brew install mingw-w64; fi
    rustup target add x86_64-pc-windows-gnu
    cargo xbuild --locked --release --target x86_64-pc-windows-gnu --no-default-features
    cp target/x86_64-pc-windows-gnu/release/openvr_api.dll "$PREFIX/"
    cp target/x86_64-pc-windows-gnu/release/bin/vrclient_x64.dll "$PREFIX/bin/"
  else
    cargo xbuild --locked --release
    cp target/release/libxrizer.dylib "$PREFIX/"
    mkdir -p "$PREFIX/bin/osx32"
    ln -s ../../libxrizer.dylib "$PREFIX/bin/osx32/vrclient.dylib"
  fi
  test -f xrizer.vrdrivermanifest && cp xrizer.vrdrivermanifest "$PREFIX/" || true
  ;;
kosmickrisp)
  brew install meson bison flex molten-vk vulkan-headers vulkan-loader llvm spirv-llvm-translator spirv-tools
  python3 -m venv "$work/venv"
  "$work/venv/bin/pip" install mako pyyaml packaging
  export PKG_CONFIG_PATH="$(brew --prefix spirv-llvm-translator)/lib/pkgconfig:$(brew --prefix spirv-tools)/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  export PATH="$(brew --prefix llvm)/bin:$work/venv/bin:$(brew --prefix bison)/bin:$(brew --prefix flex)/bin:$PATH"
  build_threshold_foveation
  meson setup "$work/build" --prefix "$PREFIX" --buildtype release -Dplatforms=macos \
    -Dvulkan-drivers=kosmickrisp -Dgallium-drivers= -Dopengl=false -Dzstd=disabled --prefer-static \
    -Dthreshold-foveation-root="$work/threshold-foveation"
  meson compile -C "$work/build"
  meson install -C "$work/build"
  test -n "$(find "$PREFIX" -name '*kosmickrisp*.json' -print -quit)"
  # Zink and the ICD come from this checkout, with separate build options.
  zink_rpath='-Wl,-rpath,@loader_path/../../../../../lib'
  meson setup "$work/zink-build" --buildtype release -Dplatforms=macos \
    -Dgallium-drivers=zink -Dvulkan-drivers= -Degl=enabled -Dgles1=disabled \
    -Dgles2=enabled -Dopengl=true -Dglx=disabled -Dgbm=disabled -Dllvm=disabled \
    -Dc_link_args="$zink_rpath" -Dcpp_link_args="$zink_rpath"
  meson compile -C "$work/zink-build"
  for directory in src/egl src/mesa/glapi/es2api src/gallium/targets/dri; do
    mkdir -p "$PREFIX/zink/$directory"
    cp -P "$work/zink-build/$directory"/lib*.dylib "$PREFIX/zink/$directory/"
  done
  cp -L "$(brew --prefix vulkan-loader)/lib/libvulkan.1.dylib" "$PREFIX/lib/"
  ;;
klepton)
  make build/klepton-ld
  cp build/klepton-ld "$PREFIX/bin/"
  ;;
wine)
  brew install bison mingw-w64 freetype gnutls vulkan-loader molten-vk
  export PATH="$(brew --prefix bison)/bin:$PATH"
  cd "$work"
  "$root/configure" --prefix="$PREFIX" --enable-archs=i386,x86_64 --disable-tests \
    --without-x --with-freetype --with-gnutls --without-gstreamer --without-sdl \
    --without-cups --without-sane --without-usb --without-v4l2 --without-pcap \
    --without-krb5 --without-gphoto --without-capi --without-netapi --without-opencl --without-ffmpeg
  make -j3
  make install
  for lib in "$(brew --prefix gnutls)/lib/libgnutls.30.dylib" \
             "$(brew --prefix freetype)/lib/libfreetype.6.dylib" \
             "$(brew --prefix vulkan-loader)/lib/libvulkan.1.dylib" \
             "$(brew --prefix molten-vk)/lib/libMoltenVK.dylib"; do cp -L "$lib" "$PREFIX/lib/"; done
  test -x "$PREFIX/bin/wine"
  ;;
dxmt)
  work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/threshold-dxmt.XXXXXX")
  brew install meson mingw-w64
  mkdir -p "$work/wine"
  curl --fail --location --retry 3 https://github.com/3Shain/wine/releases/download/v8.16-3shain/wine.tar.gz -o "$work/wine.tar.gz"
  shasum -a 256 "$work/wine.tar.gz" >> "$PREFIX/build-inputs.txt"
  tar -xzf "$work/wine.tar.gz" -C "$work/wine"
  clone_input https://github.com/llvm/llvm-project.git llvmorg-15.0.7 "$work/llvm-source"
  cmake -S "$work/llvm-source/llvm" -B "$work/llvm-build" -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$work/llvm" -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_ZSTD=OFF -DLLVM_TARGETS_TO_BUILD= \
    -DLLVM_BUILD_TOOLS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DCMAKE_CXX_FLAGS=-D_LIBCPP_KEEP_TRANSITIVE_INCLUDES_LLVM23
  cmake --build "$work/llvm-build" --parallel 3
  cmake --install "$work/llvm-build"
  meson setup "$work/build" --cross-file build-win64.txt --buildtype release --prefix "$work/install" \
    -Dnative_llvm_path="$work/llvm" -Dwine_install_path="$work/wine" -Dwine_builtin_dll=true
  meson compile -C "$work/build"
  meson install -C "$work/build"
  if [[ "$PLATFORM" == windows-x86_64 ]]; then
    cp -R "$work/install/x86_64-windows" "$PREFIX/"
    test -f "$PREFIX/x86_64-windows/d3d11.dll"
  else
    cp -R "$work/install/x86_64-unix" "$PREFIX/"
    test -f "$PREFIX/x86_64-unix/winemetal.so"
  fi
  ;;
macos-wine-xr)
  if [[ "$PLATFORM" == windows-x86_64 ]]; then
    brew install mingw-w64
    clone_input https://github.com/KhronosGroup/OpenXR-SDK.git release-1.1.63 "$work/openxr"
    cmake -S "$work/openxr" -B "$work/loader" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
      -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ -DBUILD_TESTS=OFF -DBUILD_API_LAYERS=OFF \
      -DBUILD_LOADER=ON -DDYNAMIC_LOADER=ON -DBUILD_WITH_SYSTEM_JSONCPP=OFF \
      -DCMAKE_SHARED_LINKER_FLAGS='-static-libgcc -static-libstdc++'
    cmake --build "$work/loader" --target openxr_loader --parallel 3
    loader=$(find "$work/loader" -name '*openxr_loader.dll' -print -quit)
    test -n "$loader"
    cp "$loader" "$PREFIX/openxr_loader.dll"
    for license in "$work/openxr"/LICENSE*; do test ! -f "$license" || cp "$license" "$PREFIX/licenses/"; done
  else
    zsh scripts/package-steam-play.zsh "$work/steam-play"
    cp -R "$work/steam-play/." "$PREFIX/"
  fi
  ;;
*) echo "Unsupported component: $COMPONENT" >&2; exit 2 ;;
esac
