
# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import re


class GlobalParamSection:
    """ Represent a section with parameters """
    def __init__(self, allow_duplicates=False):
        self.parameters = {}
        self.allow_duplicates = allow_duplicates
    
    def add(self, name, value):
        """ Add a parameter value """
        if self.allow_duplicates:
            self.parameters.setdefault(name, []).append(value)
        else:
            self.parameters[name] = value
    
    def set_section(self, section_dict):
        """ Replace the section with section_dict """
        self.parameters = {}
        for name, values in section_dict.items():
            for value in values if isinstance(values, list) else [values]:
                self.add(name, value)
    
    def __getitem__(self, name):
        """ Get parameter values by name """
        return self.parameters.get(name)

    def __repr__(self):
        return f"GlobalParamSection({dict(self.parameters)})"
    
class GlobalParamParser:
    """ Represent a file parser """
    def __init__(self):
        self.sections = {}
        self.section_names = []
        self.header = []
    
    def add_section(self, name):
        """ Add a new section """
        if name not in self.sections:
            allow_duplicates = True if re.match(r'^(FORCE_TYPE|DOMAIN_TYPE|OUTVAR\d*)$', name) else False
            self.sections[name] = GlobalParamSection(allow_duplicates)
            self.section_names.append(name)

    def set(self, section, name, value):
        """ Set a parameter value in a section """
        self.sections.setdefault(section, GlobalParamSection()).add(name, value)
    
    def set_section_values(self, section_name, section_dict):
        """ Replace the parameters in the overall section,
        it can allow_duplicates, i.e., "OUTVAR": {{"OUTVAR": ["OUT_RUNOFF", "OUT_BASEFLOW"]}}
        """
        self.sections.setdefault(section_name, GlobalParamSection()).set_section(section_dict)

    def get(self, section_name, param_name):
        """ Get a parameter value from a section """
        return self.sections.get(section_name, {})[param_name]
    
    def load(self, file_or_path, header_lines=5):
        """ Load the configuration from a file """
        # read
        if isinstance(file_or_path, (str, bytes)):
            file = open(file_or_path, 'r')
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
                if line == '' or (line.startswith('#') and not re.match(r'^\s*#\s*\[\s*.+?\s*\]\s*$', line)):
                    continue
                
                # identify section: #[section]
                section_match = re.match(r'^#\s*\[(.+?)\]\s*$', line)
                if section_match:
                    current_section = section_match.group(1).strip()
                    self.add_section(current_section)
                    continue
                
                # match and save into parameters
                match = re.match(r'^(\S+)\s+(.+?)(\s+#.*)?$', line)
                if match and current_section:
                    param_name = match.group(1).strip()
                    param_value = match.group(2).strip()
                    self.set(current_section, param_name, param_value)
        finally:
            if should_close:
                file.close()
    
    def write(self, file):
        """ Save the GlobalParam back to a file """
        # write header
        for line in self.header:
            file.write(line + '\n')

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
            file.write('\n')
    
    def remove_section(self, section_name):
        """ Remove a section """
        self.sections.pop(section_name, None)
        self.section_names.remove(section_name)
            
    def __getitem__(self, section):
        """ Get a section """
        return self.sections.get(section)

    def __repr__(self):
        """ Output the original format of the configuration as a string """
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