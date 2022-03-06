
import os

def generate_file_with_n_lines(n=200000,filepath="linux_dialogs",extension=".csv"):


    nfirstlines = []

    with open(filepath + extension) as f, open(filepath +"_"+str(n) + extension, "w") as out:
        for x in range(n):
            nfirstlines.append(next(f))
        for line in nfirstlines:
            out.write(line)



if __name__ == "__main__":
    generate_file_with_n_lines()