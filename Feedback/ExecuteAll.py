__author__ = 'sherlock'
import Combined
import Jaccard
import Graph
import time

if __name__ == "__main__":
    print "Started Now...................."
    start_time = time.time()
    Combined.main()
    Jaccard.main()
    Graph.main()
    print("------------ %s Total Minutes ---" % str((time.time() - start_time)/60))