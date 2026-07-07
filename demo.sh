

VIDEO="$1"

CAMERA_ROOT="${HMR_CAMERA_ROOT:-${SCENE_CAMERA_ROOT:-}}"
if [ -n "$CAMERA_ROOT" ]; then
  python tools/demo/demo.py --video="$VIDEO" --camera_root="$CAMERA_ROOT"
else
  python tools/demo/demo.py --video="$VIDEO"
fi
# python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
