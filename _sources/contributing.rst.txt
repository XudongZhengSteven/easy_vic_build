Contributing
============

Contributions are welcome through pull requests.

Workflow
--------

1. Fork the repository and create a feature branch.
2. Implement changes with clear commit messages.
3. Add or update tests and documentation when behavior changes.
4. Run local checks before opening a pull request.

Recommended local checks
------------------------

.. code-block:: bash

   python -m compileall src

If your changes affect docs:

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

Pull request checklist
----------------------

- Scope is focused and clearly described.
- Backward compatibility is considered.
- New parameters and outputs are documented.
- Example scripts are updated when relevant.
