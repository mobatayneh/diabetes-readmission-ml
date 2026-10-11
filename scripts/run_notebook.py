"""Execute a notebook with the current Python environment; retain errors in the output."""

import argparse
from pathlib import Path
import sys
import traceback

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=1800, help="Maximum seconds per cell")
    parser.add_argument("--in-process", action="store_true",
                        help="Sequential IPython execution without kernel sockets (trusted notebooks only; no per-cell timeout)")
    args = parser.parse_args(argv)
    source = args.notebook if args.notebook.is_absolute() else ROOT / args.notebook
    output = args.output or ROOT / "reports/runs" / f"{source.stem}.executed.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    if args.in_process:
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        import matplotlib
        matplotlib.use("module://matplotlib_inline.backend_inline")
        shell = InteractiveShell.instance()
        from matplotlib_inline.backend_inline import configure_inline_support
        configure_inline_support(shell, "module://matplotlib_inline.backend_inline")
        import os
        original_cwd = Path.cwd()
        os.chdir(ROOT)
        notebook.metadata["execution_method"] = "Sequential IPython in-process; no Jupyter kernel transport"
        try:
            count = 0
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                count += 1
                cell.outputs = []
                cell.execution_count = count
                # The inline backend is already configured; editor autoreload
                # magics are irrelevant to one fresh sequential process.
                source_text = "\n".join(
                    line for line in cell.source.splitlines()
                    if line.strip() not in ("%matplotlib inline", "%load_ext autoreload", "%autoreload 2")
                )
                with capture_output(display=True) as captured:
                    result = shell.run_cell(source_text, store_history=False)
                for stream, content in (("stdout", captured.stdout), ("stderr", captured.stderr)):
                    if content:
                        cell.outputs.append(nbformat.v4.new_output("stream", name=stream, text=content))
                for item in captured.outputs:
                    cell.outputs.append(nbformat.v4.new_output("display_data", data=item.data, metadata=item.metadata))
                error = result.error_before_exec or result.error_in_exec
                if error:
                    cell.outputs.append(nbformat.v4.new_output(
                        "error", ename=type(error).__name__, evalue=str(error),
                        traceback=traceback.format_exception(type(error), error, error.__traceback__),
                    ))
                    raise RuntimeError(f"Cell {count} failed: {error}") from error
                print(f"Executed code cell {count}", flush=True)
        finally:
            nbformat.write(notebook, output)
            os.chdir(original_cwd)
            print(f"Notebook output: {output}")
        return
    # The Python interpreter running this script is also used for its kernel.
    from jupyter_client.kernelspec import KernelSpec, KernelSpecManager

    class CurrentPythonKernel(KernelSpecManager):
        def get_kernel_spec(self, kernel_name):
            return KernelSpec(
                argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                display_name="Current project Python", language="python",
            )

    from jupyter_client.manager import KernelManager
    manager = KernelManager(kernel_name="python3", kernel_spec_manager=CurrentPythonKernel())
    client = NotebookClient(notebook, timeout=args.timeout, kernel_name="python3", km=manager,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
        print(f"Notebook output: {output}")
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
