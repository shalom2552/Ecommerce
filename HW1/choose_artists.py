"""
Enter your ids below (if you are submitting alone DO NOT CHANGE ID2) and execute the code.
The list of ids you get is the list of artists you need to promote.
"""

####################
# TODO: change this
ID1 = '206320772'
ID2 = '313510679'
####################


def main():
    x = (int(ID1[-1]) + int(ID2[-1])) % 5
    y = (int(ID1[-2]) + int(ID2[-2])) % 5
    options = [(70, 150), (989, 16326), (144882, 194647), (389445, 390392), (511147, 532992)]
    y = (y + 1) % 5 if x == y else y
    print("your artists are:")
    print(*options[x], *options[y])

# output:
# your artists are:
# 989 16326 511147 532992


if __name__ == '__main__':
    main()
