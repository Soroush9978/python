#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile
from bs4 import BeautifulSoup
import argparse
import re

def extract_main_content(html_file_path):
    """Extract only the main content from the Confluence HTML export."""
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    
    # Extract the main content section
    main_content = soup.find('div', {'id': 'main-content'})
    if not main_content:
        print("Error: Could not find main content section in the HTML file.")
        sys.exit(1)
    
    # Extract title and metadata
    title = ""
    metadata = ""
    
    title_element = soup.find('span', {'id': 'title-text'})
    if title_element:
        title = title_element.text.strip()
    
    metadata_element = soup.find('div', {'class': 'page-metadata'})
    if metadata_element:
        metadata = metadata_element.text.strip()
    
    # Handle tables better - simplify table structure
    tables = main_content.find_all('table')
    for table in tables:
        # Remove unnecessary classes and attributes
        for tag in table.find_all(True):
            if tag.name != 'table' and tag.name != 'tr' and tag.name != 'td' and tag.name != 'th':
                continue
            
            # Keep only essential attributes
            allowed_attrs = ['rowspan', 'colspan']
            attrs_to_remove = [attr for attr in tag.attrs if attr not in allowed_attrs]
            for attr in attrs_to_remove:
                del tag[attr]
    
    # Create simplified HTML with just what we need
    simplified_html = f"""<html>
<head><title>{title}</title></head>
<body>
<p>{metadata}</p>
{main_content}
</body>
</html>"""
    
    return simplified_html

def convert_to_markdown(html_content, output_file_path):
    """Use pandoc to convert HTML to Markdown."""
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(html_content)
        temp_html_path = temp_file.name
    
    try:
        # Run pandoc to convert to markdown
        pandoc_command = [
            'pandoc',
            '-f', 'html',
            '-t', 'gfm',  # GitHub-flavored markdown
            '--wrap=none',  # Don't wrap lines
            '--columns=1000',  # Wide columns to prevent table wrapping
            '-o', output_file_path,
            temp_html_path
        ]
        
        result = subprocess.run(pandoc_command, check=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running pandoc: {result.stderr}")
            return False
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing pandoc: {e}")
        return False
    finally:
        # Clean up the temporary file
        os.unlink(temp_html_path)

def post_process_markdown(markdown_file_path):
    """Apply final cleanup to the generated markdown file."""
    with open(markdown_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Clean up markdown file
    
    # Fix table alignment issues
    lines = content.split('\n')
    in_table = False
    table_lines = []
    non_table_lines = []
    
    # Identify and fix table sections
    for line in lines:
        if line.startswith('|') and not in_table:
            in_table = True
            table_lines = [line]
        elif in_table and line.startswith('|'):
            table_lines.append(line)
        elif in_table and not line.startswith('|'):
            # Process this table
            if len(table_lines) > 2:  # Valid table has header, separator and data
                # Ensure there's a proper separator row
                if not re.match(r'\|[-:\s|]+\|', table_lines[1]):
                    # Calculate number of columns from header
                    cols = len(table_lines[0].split('|')) - 2  # -2 for start/end pipe
                    separator = '|' + '---|' * cols
                    table_lines.insert(1, separator)
            
            # Add processed table lines to result
            non_table_lines.extend(table_lines)
            non_table_lines.append(line)
            in_table = False
            table_lines = []
        else:
            non_table_lines.append(line)
    
    # Handle case where file ends with a table
    if in_table:
        if len(table_lines) > 2:
            if not re.match(r'\|[-:\s|]+\|', table_lines[1]):
                cols = len(table_lines[0].split('|')) - 2
                separator = '|' + '---|' * cols
                table_lines.insert(1, separator)
        non_table_lines.extend(table_lines)

    content = '\n'.join(non_table_lines)
    
    # Remove HTML comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    
    # Clean up excessive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Save cleaned content
    with open(markdown_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def main():
    parser = argparse.ArgumentParser(description='Convert Confluence HTML exports to clean Markdown')
    parser.add_argument('input_file', help='Path to the HTML file exported from Confluence')
    parser.add_argument('-o', '--output', help='Output markdown file path (default: same name with .md extension)')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    
    if args.output:
        output_file = args.output
    else:
        # Replace .html extension with .md
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.md"
    
    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    # Extract the main content
    html_content = extract_main_content(input_file)
    
    # Convert to markdown
    if convert_to_markdown(html_content, output_file):
        # Apply final post-processing
        post_process_markdown(output_file)
        print(f"Conversion successful. Markdown saved to: {output_file}")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main()


