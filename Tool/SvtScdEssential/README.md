# SVT-SCD-Essential

`SvtScdEssential` is a fast standalone scene-change detection tool that runs SVT-AV1-Essential's internal pre-analysis and picture-decision logic without encoding.

It analyzes a video stream and outputs an SVT-AV1 configuration line (`ForceKeyFrames : 0f,123f,456f`) containing the resolved keyframe positions. This file can be passed directly to:
- `SvtAv1EncApp -c keyframes.cfg` to enforce pre-computed scene cuts during encoding.
- `SvtFgEssential --keyframes keyframes.cfg` so automatic film-grain estimation cleanly resets on scene boundaries.

## Usage

```bash
SvtScdEssential [input] [output.cfg] [options]
```

- **`[input]`**: Path to an input video file, or `-` / `stdin` (default: `-`).
  - Stdin, named pipes, and `.y4m` files are read via the built-in Y4M reader.
  - Other container formats (`.mkv`, `.mp4`, etc.) are decoded via **FFMS2** when built with `use-ffms2` (`-DUSE_FFMS2=ON`).
- **`[output.cfg]`**: Destination file for the `ForceKeyFrames` configuration line, or `-` for `stdout` (default: `-`).

### Quick Examples

Detect scene changes on an MKV file (using FFMS2) with automatic keyframe intervals and save to `scenes.cfg`:
```bash
SvtScdEssential input.mkv scenes.cfg
```

Pipe a Y4M stream from FFmpeg and write the config to `scenes.cfg` with a 5-second maximum keyint:
```bash
ffmpeg -v error -i input.mp4 -f yuv4mpegpipe - | SvtScdEssential - scenes.cfg --keyint 5s
```

Feed the resulting config into `SvtFgEssential` and `SvtAv1EncApp`:
```bash
SvtFgEssential --source input.mkv --table grain.tbl --strength -1 --keyframes scenes.cfg
SvtAv1EncApp -i input.mkv -b output.webm -c scenes.cfg --fgs-table grain.tbl
```

## Options

### Basic Options

| Option | Default | Description |
|--------|---------|-------------|
| `--keyint N` | `-2` | Periodic keyframe interval: `-2` = automatic (~5 seconds, aligned to mini-GOP), `-1` / `0` = off, `>0` = frame interval, `Ns` = seconds |
| `--min-keyint N` | `-1` | Minimum keyframe spacing: `-1` = automatic (multiple of mini-GOP length), `0` = off, `>0` = frame interval, `Ns` = seconds |
| `--version` | | Print version information |
| `-h`, `--help` | | Print full help information |

### Advanced Options

To match the exact mini-GOP and lookahead structure of a specific encode, you can pass the corresponding encoder structural settings:

| Option | Default | Description |
|--------|---------|-------------|
| `--frames N` | All frames | Maximum number of frames to process |
| `--fps-num N` `--fps-denom D` | Input FPS | Override input frame rate (both options must be specified together) |
| `--preset N` | `8` | Target encoder preset (`0..13`), used to resolve automatic hierarchy and SCD delay |
| `--hierarchical-levels N` | Auto | Target encoder frame hierarchy (`2..5`, default follows preset and resolution) |
| `--irefresh-type N` | `2` | Keyframe refresh type (`1` = forward / Open GOP, `2` = closed / IDR) |
| `--enable-tf N` | `1` | Target encoder temporal filtering mode (`0..3`) |

## Progress Display

When `stderr` is attached to a terminal, `SvtScdEssential` displays a live progress bar with frame count, speed (fps), elapsed time, and ETA. You can force-enable or disable progress output via the `SVT_SCD_PROGRESS=1|0` environment variable.
