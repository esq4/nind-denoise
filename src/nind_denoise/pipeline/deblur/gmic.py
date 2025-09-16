"""Deblur operation(s) for the nind-denoise pipeline."""

from __future__ import annotations

import logging

from ..base import Context, DeblurOperation, Environment, JobContext, StageError

logger = logging.getLogger(__name__)


class RLDeblur(DeblurOperation):
    """RL-deblur stage implemented via gmic.

    Matches the legacy interface expected by tests: no __init__ args and
    reads inputs from the provided Context.
    """

    def describe(self) -> str:  # pragma: no cover - description only
        return "Deblur (RL via gmic)"

    def execute(self, ctx: Context) -> None:
        if ctx.output_dir is None:
            raise StageError("output_dir not provided in Context")
        if ctx.cmd_gmic is None:
            raise StageError("cmd_gmic not provided in Context")
        if ctx.outpath is None or ctx.output_filepath is None:
            raise StageError("outpath and stage_two_output_filepath must be set")

        # Use a temporary target name to avoid issues with spaces; rename at the end
        tmp_out = ctx.outpath.with_name(ctx.outpath.stem + "__tmp.jpg")
        args = [
            str(ctx.cmd_gmic),
            str(ctx.output_filepath.name),
            "-fx_sharpen_reinhard",
            str(ctx.sigma),
            str(ctx.iteration),
            "-o_jpg",
            f"{tmp_out.name},{ctx.quality}",
        ]
        # Delegate to shared subprocess runner
        try:
            self._run_cmd(args, cwd=ctx.output_dir)
        except Exception as exc:  # be resilient to gmic issues in integration envs
            logger.warning(
                "RL-deblur failed to execute (%s); keeping exported image as-is", exc
            )
            return

        # Rename back to requested outpath (if gmic produced it); otherwise keep original
        tmp_path = ctx.output_dir / tmp_out.name
        if tmp_path.exists():
            tmp_path.replace(ctx.outpath)
        else:
            logger.warning(
                "RL-deblur did not create expected output %s; keeping exported image as-is",
                tmp_path,
            )
            return

        # Post-conditions
        self.verify(ctx)

    def verify(self, ctx: Context | None = None) -> None:
        if ctx is None or ctx.outpath is None:
            # Nothing to verify without context
            return
        if not ctx.outpath.exists():
            raise StageError(f"Deblur stage expected output missing: {ctx.outpath}")

    def execute_with_env(self, env: Environment, job_ctx: JobContext) -> None:
        """Execute RL deblur with type-safe Environment + JobContext."""
        if not env.tools.gmic:
            raise StageError("GMIC tool not available in Environment")

        input_path = job_ctx.intermediate_path or job_ctx.input_path
        output_path = job_ctx.output_path
        output_dir = job_ctx.output_dir

        # Use a temporary target name to avoid issues with spaces; rename at the end
        tmp_out = output_path.with_name(output_path.stem + "__tmp.jpg")
        args = [
            str(env.tools.gmic),
            str(input_path.name),
            "-fx_sharpen_reinhard",
            str(job_ctx.sigma),
            str(job_ctx.iterations),
            "-o_jpg",
            f"{tmp_out.name},{job_ctx.quality}",
        ]

        try:
            self._run_cmd(args, cwd=output_dir)
        except Exception as exc:
            logger.warning(
                "RL-deblur failed to execute (%s); keeping exported image as-is", exc
            )
            return

        # Rename back to requested output_path (if gmic produced it)
        tmp_path = output_dir / tmp_out.name
        if tmp_path.exists():
            tmp_path.replace(output_path)
        else:
            logger.warning(
                "RL-deblur did not create expected output %s; keeping exported image as-is",
                tmp_path,
            )
            return

        self.verify_with_env(env, job_ctx)

    def verify_with_env(self, env: Environment, job_ctx: JobContext) -> None:
        """Verify RL deblur outputs with type-safe Environment + JobContext."""
        if not job_ctx.output_path.exists():
            raise StageError(
                f"Deblur stage expected output missing: {job_ctx.output_path}"
            )


class NoOpDeblur(DeblurOperation):
    """A no-op deblur stage for disabling deblurring while preserving pipeline shape."""

    def describe(self) -> str:  # pragma: no cover - description only
        return "Deblur (no-op)"

    def execute(self, ctx: Context) -> None:
        # Intentionally do nothing; assume previous output is the final image
        if getattr(ctx, "verbose", False):
            logger.info("Deblur disabled; skipping.")
        # No verification strictness here; caller may still perform final checks

    def verify(self, ctx: Context | None = None) -> None:  # pragma: no cover - trivial
        return

    def execute_with_env(self, env: Environment, job_ctx: JobContext) -> None:
        """Execute no-op deblur with type-safe Environment + JobContext."""
        # Intentionally do nothing; assume previous output is the final image
        if env.verbose:
            logger.info("Deblur disabled; skipping.")

    def verify_with_env(self, env: Environment, job_ctx: JobContext) -> None:
        """Verify no-op deblur with type-safe Environment + JobContext."""
        # No verification needed for no-op operation
        return
