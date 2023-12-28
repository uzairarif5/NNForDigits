import os
import pickle
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))

def getImagesAndLabels():
  file = open(dir_path + "/ownLabels.txt","r")
  labels = list(map(int,(file.read()).split("\n")))
  file.close()
  file = open(dir_path + "/ownImages.dat","rb")
  images = np.array(pickle.load(file), dtype="int16")
  file.close()
  return images, labels

def saveLabelsAndImgs(images, labels):
  file = open(dir_path + "/ownImages.dat","wb")
  pickle.dump(images, file)
  file.close()
  file = open(dir_path + "/ownLabels.txt","w")
  file.write("\n".join(str(i) for i in labels))
  file.close()

def delFromIndex(iToDel):
  images, labels = getImagesAndLabels()
  print("current Label size before delete:", len(labels), "| images shape:", np.shape(images))
  images = np.delete(images, iToDel, axis=0)
  del labels[iToDel]
  print("current Label size before delete:", len(labels), "| images shape:", np.shape(images))
  saveLabelsAndImgs(images, labels)

if(__name__ == "__main__"):
  userInp = input("Press 1 to delete specific index\n")
  if(userInp == "1"):
     delFromIndex(int(input("Choose index: ")))
  else:
    print("bye")
    