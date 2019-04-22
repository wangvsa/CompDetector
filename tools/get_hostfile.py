import sys

# e.g. comet-10-[39,63], comet-10-[1,7], comet-1-3
nodelist = sys.argv[1]
#nodelist = "comet-10-[39,63], comet-10-[1,7], comet-1-3"
nodelist = ","+nodelist.replace(" ", "")

if ",comet" in nodelist:
    nodesets = nodelist.split(",comet")[1:]
    #print nodesets
    for nodeset in nodesets:
        if "[" in nodeset and "]" in nodeset:
            nodes = nodeset.split("[")[1].split("]")[0].split(",")
            basename = "comet"+nodeset.split("[")[0]
            for node in nodes:
                if "-" in node: # e.g. 48-50
                    start = int(node.split("-")[0])
                    end = int(node.split("-")[1])
                    for i in range(start, end+1):
                        print basename+(str(i) if i >= 10 else "0"+str(i))
                else:
                    print basename+node
        else:               # only has one node
            print "comet"+nodeset
else :
    nodes = nodelist.split(",")
    for node in nodes:
        print node
