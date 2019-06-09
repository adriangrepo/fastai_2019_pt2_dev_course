import os

def removeComments(inputFileName, outputFileName):

    input = open(inputFileName, "r")
    output = open(outputFileName, "w")

    output.write(input.readline())

    for line in input:
        if not line.lstrip().startswith("# In["):
            if not line.lstrip().startswith("get_ipython()"):
                output.write(line)

    input.close()
    output.close()

if __name__ == "__main__":
    print(os.listdir())
    for f in os.listdir():
        if os.path.isdir(f):
            continue
        if f != 'clean_nb_output.py'and '_clean' not in f:
            try:
                pre, post = f.split('.')
                newfile = pre+'_clean.py'
                removeComments(f,newfile)
                os.remove(f)
                os.rename(newfile,f)
            except Exception as e:
                print(f'Error with {f}: {e}')
