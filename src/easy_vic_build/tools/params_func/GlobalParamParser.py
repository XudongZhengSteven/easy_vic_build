# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: GlobalParamParser

This module provides functionality for parsing, modifying, and writing configuration files
that contain sections with parameters and their values. It includes two classes:
`GlobalParamSection` and `GlobalParamParser`. The `GlobalParamSection` class represents a
single section of the configuration file and handles storing parameters and their values,
while the `GlobalParamParser` class manages the entire configuration file, allowing for
section and parameter manipulation, loading from files, and writing back to files.

Class:
------
    - GlobalParamSection: A class that represents a section in the configuration file,
      allowing for the addition, retrieval, and organization of parameters.

    - GlobalParamParser: A class that handles the parsing of the entire configuration file,
      manages sections and their parameters, and supports reading from and writing to files.

Class Methods:
--------------
    - __init__: Initializes the `GlobalParamParser` instance.
    - add_section: Adds a new section to the configuration if it doesn't already exist.
    - set: Sets a parameter value within a specific section.
    - set_section_values: Replaces the parameters of a section with new values from a dictionary.
    - get: Retrieves a parameter value from a specific section.
    - load: Loads the configuration from a file, parsing sections and parameters.
    - write: Writes the current configuration back to a file.
    - remove_section: Removes a section from the configuration.
    - __getitem__: Retrieves a section by name.
    - __repr__: Returns a string representation of the entire configuration.

Dependencies:
-------------
    - re: Used for regular expression matching to identify sections and parameters.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com
"""


import re


class GlobalParamSection:
    """
    Represents a section with parameters in a configuration.

    A section is a collection of key-value pairs, where keys are the parameter names
    and values are their associated values. This class provides functionality to add
    parameters, replace the section with a new set of parameters, and retrieve
    parameter values.

    Attributes
    ----------
    parameters : dict
        A dictionary that stores parameter names as keys and their corresponding values.
    allow_duplicates : bool
        A flag indicating whether duplicate parameter names are allowed.

    Methods
    -------
    add(name, value)
        Adds a parameter value to the section. If duplicates are allowed, appends
        the value to a list, otherwise replaces the existing value.
    set_section(section_dict)
        Replaces the current parameters with those from the provided dictionary.
    __getitem__(name)
        Retrieves the parameter values associated with the specified name.
    __repr__()
        Returns a string representation of the section with its parameters.
    """

    def __init__(self, allow_duplicates=False):
        """
        Initializes the GlobalParamSection instance.

        Parameters
        ----------
        allow_duplicates : bool, optional
            A flag indicating whether duplicate parameter names are allowed (default is False).
        """
        self.parameters = {}
        self.allow_duplicates = allow_duplicates

    def add(self, name, value):
        """
        Adds a parameter value to the section.

        Parameters
        ----------
        name : str
            The name of the parameter.
        value : str
            The value associated with the parameter.

        Notes
        -----
        If duplicates are allowed, the value will be appended to a list. Otherwise,
        it will replace any existing value for the given name.
        """
        if self.allow_duplicates:
            self.parameters.setdefault(name, []).append(value)
        else:
            self.parameters[name] = value

    def set_section(self, section_dict):
        """
        Replaces the section's parameters with those in the provided dictionary.

        Parameters
        ----------
        section_dict : dict
            A dictionary where keys are parameter names and values are their corresponding values.
            The values can either be a single value or a list of values.

        Notes
        -----
        This method will completely replace the section's current parameters with the new values.
        """
        self.parameters = {}
        for name, values in section_dict.items():
            for value in values if isinstance(values, list) else [values]:
                self.add(name, value)

    def __getitem__(self, name):
        """
        Retrieves the parameter values associated with the given name.

        Parameters
        ----------
        name : str
            The name of the parameter.

        Returns
        -------
        list or str
            The parameter values associated with the specified name.
            Returns None if the parameter is not found.
        """
        return self.parameters.get(name)

    def __repr__(self):
        """
        Returns a string representation of the section with its parameters.

        Returns
        -------
        str
            A string representation of the GlobalParamSection, showing its parameters.
        """
        return f"GlobalParamSection({dict(self.parameters)})"


class GlobalParamParser:
    """
    A parser for reading and writing configuration files.

    This class provides functionality for parsing configuration files into
    sections with parameters and values. It supports reading configuration
    files, modifying parameters, and writing the configuration back to files.
    Each section can contain multiple parameters, and the parser handles
    setting, retrieving, and organizing these parameters.

    Attributes
    ----------
    sections : dict
        A dictionary where keys are section names and values are `GlobalParamSection` instances
        that hold parameters for that section.
    section_names : list
        A list of section names in the configuration.
    header : list
        A list of header lines in the configuration file.

    Methods
    -------
    add_section(name)
        Adds a new section to the parser if it doesn't already exist.
    set(section, name, value)
        Sets a parameter value in a given section.
    set_section_values(section_name, section_dict)
        Replaces the parameters of a section with new values from a dictionary.
    get(section_name, param_name)
        Retrieves a parameter value from a specific section.
    load(file_or_path, header_lines=5)
        Loads a configuration from a file, parsing it into sections and parameters.
    write(file)
        Writes the current configuration back to a file.
    remove_section(section_name)
        Removes a section from the configuration.
    __getitem__(section)
        Retrieves a section by name.
    __repr__()
        Returns a string representation of the configuration in its original format.
    """

    def __init__(self):
        """
        Initializes the GlobalParamParser instance.

        This constructor sets up empty containers for sections, section names, and header lines.
        """
        self.sections = {}
        self.section_names = []
        self.header = []

    def add_section(self, name):
        """
        Adds a new section to the configuration parser.

        Parameters
        ----------
        name : str
            The name of the section to be added.

        Notes
        -----
        If the section already exists, no action is taken. If the section name matches
        a specific pattern (e.g., 'FORCE_TYPE', 'DOMAIN_TYPE', etc.), duplicates are allowed in that section.
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
        """
        Sets a parameter value in a specified section.

        Parameters
        ----------
        section : str
            The name of the section in which the parameter will be set.
        name : str
            The name of the parameter.
        value : str
            The value of the parameter.

        Notes
        -----
        If the section does not exist, it will be created automatically.
        """
        self.sections.setdefault(section, GlobalParamSection()).add(name, value)

    def set_section_values(self, section_name, section_dict):
        """
        Replaces the parameters in a section with those from a provided dictionary.

        Parameters
        ----------
        section_name : str
            The name of the section to be modified.
        section_dict : dict
            A dictionary containing parameter names as keys and their corresponding values.

        Notes
        -----
        This method allows for setting parameters, including duplicate values,
        in sections like 'OUTVAR'.
        it can allow_duplicates, i.e., "OUTVAR": {{"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW"]}}
        """
        self.sections.setdefault(section_name, GlobalParamSection()).set_section(
            section_dict
        )

    def get(self, section_name, param_name):
        """
        Retrieves a parameter value from a specified section.

        Parameters
        ----------
        section_name : str
            The name of the section from which the parameter will be retrieved.
        param_name : str
            The name of the parameter.

        Returns
        -------
        str
            The value of the parameter, or None if the parameter is not found.
        """
        return self.sections.get(section_name, {})[param_name]

    def load(self, file_or_path, header_lines=5):
        """
        Loads a configuration from a file and parses it into sections and parameters.

        Parameters
        ----------
        file_or_path : str or file-like object
            The path to the configuration file or a file-like object to read from.
        header_lines : int, optional
            The number of header lines to skip before parsing the sections (default is 5).

        Notes
        -----
        If the file is passed as a path, it will be opened. If it's a file-like object,
        it will be read directly. The file should be in a specific format with sections
        indicated by lines starting with '# ['.
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
        """
        Writes the current configuration to a file.

        Parameters
        ----------
        file : file-like object
            The file-like object where the configuration will be written.

        Notes
        -----
        The file will be written in the same format as the original configuration.
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
        """
        Removes a section from the configuration.

        Parameters
        ----------
        section_name : str
            The name of the section to be removed.
        """
        self.sections.pop(section_name, None)
        self.section_names.remove(section_name)

    def __getitem__(self, section):
        """
        Retrieves a section by name.

        Parameters
        ----------
        section : str
            The name of the section.

        Returns
        -------
        GlobalParamSection
            The section object associated with the given name.
        """
        return self.sections.get(section)

    def __repr__(self):
        """
        Returns a string representation of the configuration in its original format.

        Returns
        -------
        str
            The string representation of the entire configuration, including header and sections.
        """
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
