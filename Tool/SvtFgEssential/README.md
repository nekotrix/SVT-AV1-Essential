# SVT-FG-Essential

`SvtFgEssential` is a standalone AV1 film-grain estimation and synthesis tool powered by SVT-AV1-Essential's internal noise modeling (`noise_model.c`), automatic film-grain analysis (`film_grain_analysis.c`), and grain synthesis (`grainSynthesis.c`).

It supports three primary workflows:
1. **Automatic grain analysis (`--strength -1`) (Best)**: Dynamically estimates film-grain parameters directly from the noisy source without manual strength guessing or pre-denoising.
2. **Manual internal denoising (`--strength 1..50`)**: Denoises the source at the chosen strength and models the removed noise as AV1 film grain.
3. **External denoised reference (`--target FILE`) (WIP)**: Estimates AV1 film grain from the difference between a noisy `--source` and your own externally denoised `--target` video.

At least one output must be specified: `--table` (AV1 film-grain table), `--denoised-output` (denoised Y4M/YUV video), or `--output` (Y4M/YUV video with synthesized AV1 grain applied for previewing).

## Usage

```bash
SvtFgEssential --source <input> --table <output.tbl> [options]
```

### Quick Examples

**1. Automatic grain table generation (recommended with `SvtScdEssential` keyframes):**
```bash
SvtScdEssential input.mkv scenes.cfg
SvtFgEssential --source input.mkv --table grain.tbl --strength -1 --keyframes scenes.cfg
```

**2. Manual strength estimation with additional parallel worker threads and the optimized seed sequence:**
```bash
SvtFgEssential --source input.y4m --table grain.tbl --strength 10 --seed -1 --jobs 4 --fast
```

**3. Estimate grain between a noisy source and an externally denoised video (WIP):**
```bash
Xdenoiser -i noisy.mkv -o - | SvtFgEssential --source noisy.mkv --target - --table grain.tbl --jobs 4
```

## Options

### Input and Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source FILE` (`--input`, `-i`) | *(Required)* | Noisy source video. Supports 8/10-bit 4:2:0 Y4M, planar YUV, `-`/`stdin`/pipes, or any container via FFMS2 (when built with `use-ffms2`) |
| `--target FILE` (`--denoised-input`) | None | Previously denoised video of matching dimensions and bit depth; grain is estimated from the difference between `--source` and `--target` |
| `--table FILE` | None | Output path for the AV1 film-grain table (compatible with `--fgs-table`) |
| `--output FILE` (`-o`) | None | Output path for video with synthesized grain applied (`.y4m` writes Y4M; other extensions write raw planar YUV) |
| `--denoised-output FILE` | None | Output path for the denoised video (`.y4m` writes Y4M; other extensions write raw planar YUV) |
| `--frames N` (`-n`) | All frames | Maximum number of frames to process |
| `--width W` (`-w`), `--height H` (`-hgt`) | From header | Frame dimensions (required for raw YUV input; must be even) |
| `--bit-depth 8\|10` | `8` | Raw YUV input bit depth |
| `--fps-num N` `--fps-denom D` | Input / `24/1` | Frame rate numerator and denominator (specify both together) |
| `--source-format auto\|raw\|ffms2` | `auto` | Force reader backend for `--source` (at most one of `--source` or `--target` may use FFMS2 or a pipe) |
| `--target-format auto\|raw\|ffms2` | `auto` | Force reader backend for `--target` |

### Grain Options

| Option | Default | Description |
|--------|---------|-------------|
| `--strength -1\|1..50` (`--noise-level`) | `6` | Denoising strength (`1..50`), or `-1` for automatic grain estimation |
| `--multiplier FACTOR` | `1` | Grain amplitude multiplier for automatic analysis without `--target` (`0..2`; `0` disables synthesis) |
| `--adaptive 0\|1` | `0` | Adapt denoising block size to video resolution (applies to manual `--strength` or `--target`) |
| `--jobs N` | `2` | Number of frames to analyze in parallel (`1..256`; applies to manual `--strength` or `--target`) |
| `--seed S` | Per-frame | Fixed grain-pattern seed (`1..65535`), or `-1` to cycle through a tested sequence that reduces repeating grain patterns |
| `--seeds S1,S2,...\|-1` | None | Cycle through a custom comma-separated seed list (`1..65535`), or `-1` for the tested optimized seed sequence |
| `--fast` | Off | Enable additional optimizations to the --strength 1..50 path without altering results |

### Keyframe and Table Timing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--keyint N` | Auto | Periodic keyframe interval in frames (`0` disables periodic keyframes; default follows frame rate and encoder hierarchy) |
| `--keyframes LIST\|FILE\|@FILE` | None | Comma-separated frame numbers, a plain frame list file, or an `SvtScdEssential` config file (`@FILE` forces file reading). Automatic analysis resets on these frames |
| `--preset N` | `8` | Encoder preset (`0..13`) used to derive automatic keyframe spacing |
| `--hierarchical-levels N` | Auto | Encoder frame hierarchy (`2..5`) used for keyframe and table update spacing |
| `--irefresh-type N` | `2` | Encoder keyframe type (`1` = forward, `2` = closed) for table timing alignment |
| `--table-events svt-ra\|per-frame` (`--table-event-mode`) | `svt-ra` | Write grain-table updates on SVT-AV1's Random Access schedule (`svt-ra`) or on every frame (`per-frame`; always `per-frame` in automatic mode) |
| `--version` | | Print version information |
| `-h`, `--help` | | Print full help information |
