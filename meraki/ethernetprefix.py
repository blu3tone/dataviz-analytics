import json

'''
Creates a json database of Ethernet device manufacturers, indexed by 
Ethernet MAC prefix.

The source OUI file can be downloaed from the IEEE at 
http://standards-oui.ieee.org/oui/oui.txt
'''


def loadOuiFile(filename=None):
    
    if filename==None:
        filename='data/oui.txt'
        
    prefixes={}    
        
    for line in open(filename,"r") :
        if line.find("(hex)") != -1:
            fields = line.split()
            prefix = fields[0].replace('-',':').lower()
            who = ' '.join(fields[2:])
            prefixes[prefix]=who
            
    with open('data/prefixes.json','w') as jsonfile:
        json.dump(prefixes, jsonfile,indent=3, sort_keys=True)   
               
loadOuiFile()