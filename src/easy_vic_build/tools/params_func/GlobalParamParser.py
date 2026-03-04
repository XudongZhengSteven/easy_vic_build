# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Parser utilities for VIC global parameter files."""


import re


class GlobalParamSection:
    """Container for parameters within one global-parameter section."""

    def __init__(self, allow_duplicates=False):
        """Initialize a section container.

        Parameters
        ----------
        allow_duplicates : bool, optional
            Whether repeated parameter names should be stored as lists.
        """
        self.parameters = {}
        self.allow_duplicates = allow_duplicates

    def add(self, name, value):
        """Add one parameter entry.

        Parameters
        ----------
        name : str
            Parameter name.
        value : str
            Parameter value.

        Notes
        -----
        If duplicates are enabled, values are appended as a list; otherwise the
        latest value overwrites previous value.
        """
        if self.allow_duplicates:
            self.parameters.setdefault(name, []).append(value)
        else:
            self.parameters[name] = value

    def set_section(self, section_dict):
        """Replace all parameters in the section.

        Parameters
        ----------
        section_dict : dict
            Mapping from parameter name to a single value or list of values.

        Notes
        -----
        Existing parameters are cleared before assignment.
        """
        self.parameters = {}
        for name, values in section_dict.items():
            for value in values if isinstance(values, list) else [values]:
                self.add(name, value)

    def __getitem__(self, name):
        """Get parameter value(s) by name.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        str or list or None
            Stored value(s), or ``None`` if name is missing.
        """
        return self.parameters.get(name)

    def __repr__(self):
        """Return debug representation of section parameters."""
        return f"GlobalParamSection({dict(self.parameters)})"


class GlobalParamParser:
    """Parser for VIC global-parameter text files."""

    def __init__(self):
        """Initialize an empty parser state."""
        self.sections = {}
        self.section_names = []
        self.header = []

    def add_section(self, name):
        """Add a section if it does not already exist.

        Parameters
        ----------
        name : str
            Section name.

        Notes
        -----
        Duplicate keys are enabled for sections matching
        ``FORCE_TYPE``, ``DOMAIN_TYPE``, and ``OUTVAR*``.
        """
        if name not in self.sections:
            allow_duplicates = (
                True
                if re.match(r"^(FORCE_TYPE|DOMAIN_TYPE|OUTVAR\d*)$", name)
                else False
            )
            self.sections[name] = GlobalParamSection(allow_duplicates)
            self.section_names.append(name)

    def set(self, section, name, value):
        """Set one parameter value in a section.

        Parameters
        ----------
        section : str
            Section name.
        name : str
            Parameter name.
        value : str
            Parameter value.

        Notes
        -----
        Section is created automatically if missing.
        """
        self.sections.setdefault(section, GlobalParamSection()).add(name, value)

    def set_section_values(self, section_name, section_dict):
        """Replace all parameters in a section.

        Parameters
        ----------
        section_name : str
            Section name.
        section_dict : dict
            Mapping from parameter name to value or list of values.

        Notes
        -----
        Existing parameters in this section are replaced.
        """
        self.sections.setdefault(section_name, GlobalParamSection()).set_section(
            section_dict
        )

    def get(self, section_name, param_name):
        """Get one parameter value from a section.

        Parameters
        ----------
        section_name : str
            Section name.
        param_name : str
            Parameter name.

        Returns
        -------
        str
            Parameter value.

        Raises
        ------
        KeyError
            If section or parameter is missing.
        """
        return self.sections.get(section_name, {})[param_name]

    def load(self, file_or_path, header_lines=5):
        """Load and parse global-parameter text.

        Parameters
        ----------
        file_or_path : str or file-like object
            File path or readable file-like object.
        header_lines : int, optional
            Number of header lines to store before section parsing.

        Raises
        ------
        ValueError
            If input is neither path-like nor file-like.
        """
        # read
        if isinstance(file_or_path, (str, bytes)):
            file = open(file_or_path, "r")
            should_close = True
        elif hasattr(file_or_path, "read"):
            file = file_or_path
            should_close = False
        else:
            raise ValueError("file_or_path must be a file path or a file-like object")

        # read and parse
        # with open(filepath, 'r') as file:
        try:
            for _ in range(header_lines):
                self.header.append(file.readline().strip())

            current_section = None
            for line in file:
                line = line.strip()

                # ignore space lines and #
                if line == "" or (
                    line.startswith("#")
                    and not re.match(r"^\s*#\s*\[\s*.+?\s*\]\s*$", line)
                ):
                    continue

                # identify section: #[section]
                section_match = re.match(r"^#\s*\[(.+?)\]\s*$", line)
                if section_match:
                    current_section = section_match.group(1).strip()
                    self.add_section(current_section)
                    continue

                # match and save into parameters
                match = re.match(r"^(\S+)\s+(.+?)(\s+#.*)?$", line)
                if match and current_section:
                    param_name = match.group(1).strip()
                    param_value = match.group(2).strip()
                    self.set(current_section, param_name, param_value)
        finally:
            if should_close:
                file.close()

    def write(self, file):
        """Write parser content to a file-like object.

        Parameters
        ----------
        file : file-like object
            Writable file-like object.
        """
        # write header
        for line in self.header:
            file.write(line + "\n")

        # write section content
        for section_name in self.section_names:
            section = self.sections[section_name]
            file.write(f"# [{section_name}]\n")
            for key, value in section.parameters.items():
                if isinstance(value, list):
                    for v in value:
                        file.write(f"{key}\t{v}\n")
                else:
                    file.write(f"{key}\t{value}\n")
            file.write("\n")

    def remove_section(self, section_name):
        """Remove a section from parser state.

        Parameters
        ----------
        section_name : str
            Section name.

        Raises
        ------
        ValueError
            If section name is not present in internal section order list.
        """
        self.sections.pop(section_name, None)
        self.section_names.remove(section_name)

    def __getitem__(self, section):
        """Get section object by name.

        Parameters
        ----------
        section : str
            Section name.

        Returns
        -------
        GlobalParamSection or None
            Matching section object, or ``None`` if missing.
        """
        return self.sections.get(section)

    def __repr__(self):
        """Render parser content to global-parameter text format."""
        output = self.header + [""]

        for section_name in self.section_names:
            output.append(f"# [{section_name}]")
            section = self.sections[section_name]
            for key, value in section.parameters.items():
                if isinstance(value, list):
                    output.extend(f"{key}\t{v}" for v in value)
                else:
                    output.append(f"{key}\t{value}")
            output.append("")

        text = "\n".join(output)
        return text
