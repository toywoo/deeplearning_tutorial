import re
import os
import glob
import ast

pattern = re.compile(r'\[([^:]+):L(\d+)-L(\d+)\]\(file:///[^)]+\)')

md_files = glob.glob('*_expanded.md')
for md_file in md_files:
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = []
    lines = content.split('\n')
    changed = False
    
    for line in lines:
        new_content.append(line)
        matches = list(pattern.finditer(line)) # if there are multiple matches
        for match in matches:
            notebook = match.group(1)
            start_line = int(match.group(2))
            end_line = int(match.group(3))
            
            try:
                with open(notebook, 'r', encoding='utf-8') as nb:
                    nb_lines = nb.readlines()
                
                code_lines = nb_lines[start_line-1:end_line]
                clean_codes = []
                for cl in code_lines:
                    cl_strip = cl.strip()
                    if cl_strip.endswith(','):
                        cl_strip = cl_strip[:-1]
                    try:
                        parsed_str = ast.literal_eval(cl_strip)
                        clean_codes.append(parsed_str)
                    except:
                        # Fallback parsing
                        temp = cl_strip.strip('"')
                        if temp.endswith('\\n'):
                            temp = temp[:-2] + '\n'
                        temp = temp.replace('\\"', '"')
                        clean_codes.append(temp)
                
                # Append the parsed code block right below the line containing the link
                code_block = "\n```python\n" + "".join(clean_codes).rstrip('\n') + "\n```"
                new_content.append(code_block)
                changed = True
            except Exception as e:
                print(f"Error on {notebook} {start_line}-{end_line}: {e}")
                
    if changed:
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_content))
            print(f"Updated {md_file}")
