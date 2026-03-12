"""CLI entry points for bitnet2lut pipeline.

Usage:
    bitnet2lut extract --model <path> --output-dir <dir>
    bitnet2lut tile --input <dir> --output-dir <dir>
    bitnet2lut generate-luts --input <dir> --output-dir <dir>
    bitnet2lut export-fpga --input <dir> --output-dir <dir>
    bitnet2lut verify --output-dir <dir>
    bitnet2lut run-all --model <path> --output-dir <dir>
"""

import click

from .utils import setup_logging


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """bitnet2lut: Hardware-agnostic LUT conversion for BitNet b1.58 models."""
    setup_logging(verbose)


@main.command()
@click.option(
    "--model", "-m",
    default="microsoft/bitnet-b1.58-2B-4T-bf16",
    help="HuggingFace repo ID or local path to BF16 model",
)
@click.option(
    "--output-dir", "-o",
    default="outputs",
    help="Output directory",
)
@click.option(
    "--layers",
    default=None,
    help="Comma-separated layer indices (e.g., '0,1,2'). Default: all.",
)
@click.option(
    "--save-stats/--no-save-stats",
    default=True,
    help="Save per-weight statistics",
)
def extract(model: str, output_dir: str, layers: str | None, save_stats: bool) -> None:
    """Step 1: Extract and quantize weights to ternary {-1, 0, +1}."""
    from .extract import extract_weights

    layer_list = [int(x) for x in layers.split(",")] if layers else None
    extract_weights(model, output_dir, layer_list, save_stats)


@main.command()
@click.option(
    "--input", "-i",
    required=True,
    help="Directory containing ternary_weights/ from Step 1",
)
@click.option(
    "--output-dir", "-o",
    default="outputs/tiles",
    help="Output directory for tiled weights",
)
@click.option(
    "--block-size", "-b",
    default=128,
    type=int,
    help="Block size (square tiles). Default: 128",
)
@click.option(
    "--block-rows",
    default=None,
    type=int,
    help="Override block height (default: same as block-size)",
)
@click.option(
    "--block-cols",
    default=None,
    type=int,
    help="Override block width (default: same as block-size)",
)
@click.option(
    "--pad-strategy",
    type=click.Choice(["zero", "none"]),
    default="zero",
    help="Edge tile padding strategy",
)
def tile(
    input: str,
    output_dir: str,
    block_size: int,
    block_rows: int | None,
    block_cols: int | None,
    pad_strategy: str,
) -> None:
    """Step 2: Tile weight matrices into BRAM-sized blocks."""
    from .tile import tile_all_weights

    br = block_rows or block_size
    bc = block_cols or block_size
    tile_all_weights(input, output_dir, br, bc, pad_strategy)


@main.command("generate-luts")
@click.option(
    "--input", "-i",
    required=True,
    help="Directory containing tiles/ from Step 2",
)
@click.option(
    "--output-dir", "-o",
    default="outputs/luts",
    help="Output directory for LUT indices",
)
@click.option(
    "--group-size", "-g",
    default=4,
    type=int,
    help="Ternary weights per LUT group. Default: 4 (81 configs)",
)
@click.option(
    "--lut-dtype",
    default="int16",
    help="Data type for runtime LUT entries",
)
def generate_luts(
    input: str, output_dir: str, group_size: int, lut_dtype: str
) -> None:
    """Step 3: Generate T-MAC style LUT indices from tiled weights."""
    from .lut_gen import generate_luts_for_all_tiles

    generate_luts_for_all_tiles(input, output_dir, group_size, lut_dtype)


@main.command("export-fpga")
@click.option(
    "--input", "-i",
    required=True,
    help="Directory containing lut_indices/ from Step 3",
)
@click.option(
    "--tiling-map",
    required=True,
    help="Path to tiling_map.json from Step 2",
)
@click.option(
    "--lut-summary",
    required=True,
    help="Path to lut_summary.json from Step 3",
)
@click.option(
    "--output-dir", "-o",
    default="outputs/fpga",
    help="Output directory for FPGA files",
)
@click.option(
    "--format", "-f",
    multiple=True,
    default=["coe", "mem", "sv"],
    help="Export formats (can specify multiple)",
)
@click.option(
    "--hw-bit-width",
    default=8,
    type=int,
    help="Bit width per index in hardware files",
)
def export_fpga_cmd(
    input: str,
    tiling_map: str,
    lut_summary: str,
    output_dir: str,
    format: tuple[str, ...],
    hw_bit_width: int,
) -> None:
    """Step 4: Export LUT indices as FPGA BRAM init files."""
    from .export_fpga import export_fpga

    export_fpga(input, tiling_map, lut_summary, output_dir, list(format), hw_bit_width)


@main.command()
@click.option(
    "--output-dir", "-o",
    default="outputs",
    help="Pipeline output directory (must contain ternary_weights/)",
)
@click.option(
    "--group-size", "-g",
    default=4,
    type=int,
    help="LUT group size",
)
@click.option(
    "--num-vectors",
    default=5,
    type=int,
    help="Number of random vectors for Level 2 verification",
)
@click.option(
    "--layers",
    default=None,
    help="Comma-separated layer indices to verify",
)
def verify(output_dir: str, group_size: int, num_vectors: int, layers: str | None) -> None:
    """Step 5: Verify numerical correctness of the conversion pipeline."""
    from .verify import run_all_verification

    layer_list = [int(x) for x in layers.split(",")] if layers else None
    results = run_all_verification(output_dir, group_size, num_vectors, layer_list)
    if not results["all_pass"]:
        raise click.ClickException("Verification FAILED — see details above")


@main.command("run-all")
@click.option(
    "--model", "-m",
    default="microsoft/bitnet-b1.58-2B-4T-bf16",
    help="HuggingFace repo ID or local path to BF16 model",
)
@click.option(
    "--output-dir", "-o",
    default="outputs",
    help="Output directory for all pipeline artifacts",
)
@click.option(
    "--block-size", "-b",
    default=128,
    type=int,
    help="BRAM tile block size",
)
@click.option(
    "--group-size", "-g",
    default=4,
    type=int,
    help="T-MAC LUT group size",
)
@click.option(
    "--layers",
    default=None,
    help="Comma-separated layer indices (default: all 30)",
)
@click.option(
    "--skip-verify",
    is_flag=True,
    help="Skip verification step",
)
def run_all(
    model: str,
    output_dir: str,
    block_size: int,
    group_size: int,
    layers: str | None,
    skip_verify: bool,
) -> None:
    """Run the complete Phase 0 pipeline (Steps 1-5)."""
    import logging

    from .export_fpga import export_fpga
    from .extract import extract_weights
    from .lut_gen import generate_luts_for_all_tiles
    from .tile import tile_all_weights
    from .verify import run_all_verification

    logger = logging.getLogger("bitnet2lut")
    layer_list = [int(x) for x in layers.split(",")] if layers else None

    logger.info("=" * 60)
    logger.info("bitnet2lut — Full Pipeline")
    logger.info("=" * 60)

    # Step 1: Extract
    logger.info("\n[Step 1/5] Extracting and quantizing weights...")
    extract_weights(model, output_dir, layer_list, save_stats=True)

    # Step 2: Tile
    logger.info("\n[Step 2/5] Tiling weight matrices...")
    weights_dir = f"{output_dir}/ternary_weights"
    tiles_output = f"{output_dir}/tiles"
    tile_all_weights(weights_dir, tiles_output, block_size, block_size)

    # Step 3: Generate LUTs
    logger.info("\n[Step 3/5] Generating LUT indices...")
    lut_output = f"{output_dir}/luts"
    generate_luts_for_all_tiles(tiles_output, lut_output, group_size)

    # Step 4: Export FPGA files
    logger.info("\n[Step 4/5] Exporting FPGA BRAM init files...")
    tiling_map = f"{tiles_output}/tiling_map.json"
    lut_summary = f"{lut_output}/lut_summary.json"
    fpga_output = f"{output_dir}/fpga"
    export_fpga(lut_output, tiling_map, lut_summary, fpga_output)

    # Step 5: Verify
    if not skip_verify:
        logger.info("\n[Step 5/5] Running verification...")
        results = run_all_verification(output_dir, group_size, layers=layer_list)
        if not results["all_pass"]:
            raise click.ClickException("Verification FAILED")
    else:
        logger.info("\n[Step 5/5] Skipped (--skip-verify)")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"All outputs in: {output_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
