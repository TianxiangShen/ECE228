from __future__ import print_function
import urllib2
import sys
from network_elements import Mapping
import os
import os.path

def prepModel(modelfile):
    m19 = modelfile['model.19.weight'];
    im_size = m19.size(0);
    F = Mapping(3,3,im_size);
    F.cuda();
    F.load_state_dict(modelfile);
    F.eval();
    return F;

def downloadModel(modelname, mdir):
    modelurl = "https://github.com/nathanrgodwin/GAN-Style-Transfer/raw/master/model/"
    if not os.path.exists("./modelDL/"+modelname+"/"+mdir+".data"):
        try:
            if not os.path.exists("./modelDL/"+modelname):
                os.makedirs("./modelDL/"+modelname);
            url = modelurl + modelname + "/" + mdir + ".data";
            file = "./modelDL/" + modelname + "/" + mdir + ".data";
            u = urllib2.urlopen(url);
            f = open(file, 'wb');
            file_size = int(u.info().getheaders("Content-Length")[0]);

            file_size_dl = 0;
            block_sz = 8192;

            while True:
                buff = u.read(block_sz);
                if not buff:
                    break;

                file_size_dl += len(buff);
                f.write(buff);
                if ((file_size_dl * 100. /file_size)%10 == 0): 
                    print(str(int(file_size_dl * 100. / file_size))+"%")

            f.close()
            print("Downloaded model ("+modelname+")")
        except:
            print("Error downloading model " + modelname, file=sys.stderr)
        