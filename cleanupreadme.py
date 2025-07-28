import pypandoc
import re

# Convert and clean
z = pypandoc.convert_file(
    'README.md',
    'rst',
    format='markdown',
    extra_args=['--wrap=none']
)

# Replace 3 or more consecutive newlines with exactly 2 newlines
z = re.sub(r'\n{3,}', '\n\n', z)

# Then replace any 2 or more consecutive blank lines with exactly 2 newlines
# (This is a safety net; you can comment this line out if unnecessary)
z = re.sub(r'(\n\s*){3,}', '\n\n', z)

# Optionally, trim leading/trailing whitespace on each line to avoid "empty lines" with spaces/tabs
lines = z.splitlines()
lines = [line.rstrip() for line in lines]
z_clean = '\n'.join(lines)

with open('README.rst', 'w', encoding='utf-8') as outfile:
    outfile.write(z_clean)