
EXAMPLE="""
|                   |                   **Parameter Description**                          |       **Name**      |      **Value**       |    **Unit**     |                  _comment_                      |
|-------------------|----------------------------------------------------------------------|---------------------|----------------------|-----------------|-------------------------------------------------|
| **all**           |                                                                      |                     |                      |                 |                                                 |
|                   | time constant for smoothing in the Gaussian filter                   | `Tsmoothing`        | 1.32 10<sup>-4</sup> | ms              |                                                 |
| **Figure 1**      |                                                                      |                     |                      |                 |                                                 |
|                   | a threshold used in Fig 1                                            | `Fig1_threshTheta`  | 18.3                 | N/A             |                                                 |
| **Figure 2**      |                                                                      |                     |                      |                 |                                                 |
|                   | a threshold used in Fig 2                                            | `Fig2_threshTheta`  | 32.7 10<sup>-4</sup> | N/A             |                                                 |
"""

def format_key(value):
    if '10<sup>' in value:
        return float(value.replace(' 10<sup>', 'e').replace('</sup>', ''))
    elif '.' in value:
        return float(value)
    elif '[' in value:
        print('not implemented yet')
        return ()
    else:
        return int(value)

def read_table(filename,
               restrict_to_section=None):

    table = {}

    with open(filename, 'r') as f:

        for line in f.readlines():
            split = line.split('| `')
            if len(split)>1:
                key = split[1].split('`')[0]
                value = split[1].split('| ')[1].split(' |')[0]
                table[key] = format_key(value)

    return table

if __name__=='__main__':

    import tempfile, os
    with open(os.path.join(tempfile.gettempdir(), 'table.md'), 'w') as f:
        f.write(EXAMPLE)

    table = read_table(os.path.join(tempfile.gettempdir(), 'table.md'))
    print(table)

