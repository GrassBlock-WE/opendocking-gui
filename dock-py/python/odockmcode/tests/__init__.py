"""End-to-end tests for the Open Docking Python stack.

The suite ships inside the installed package, so run it against the wheel:

    python -m pytest --pyargs odockmcode -q

To write the synthetic test structures out to a directory (for poking at the CLI
by hand):

    python -m odockmcode.tests.make_data <directory>
"""
