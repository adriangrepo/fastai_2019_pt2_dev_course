import os

def write_code():
    print(os.listdir())
    output = open('nb_all.py', "w")
    for f in sorted(os.listdir()):
        if f not in ['nb_all.py', '__init__.py', 'nb_append.py']:
            if f.endswith('.py'):
                input = open(f, "r")
                output.write('\n')
                output.write('########################')
                output.write(f'# {f}')
                output.write('\n')
                i=0
                for line in input:
                    if not line.lstrip().startswith("# file"):
                        if not line.lstrip().startswith("from exp.nb_"):
                            if not line.lstrip().startswith("##"):
                                output.write(line)
                                i+=1
                print(f'wrote {i} lines from {f}')
                input.close()
                # os.remove(f)
    output.close()

if __name__ == "__main__":
    write_code()
    print('done')

