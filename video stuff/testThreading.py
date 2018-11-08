import threading 
  
def printName(i): 
    print '{}'.format(i)
  
if __name__ == "__main__": 


	for i in range(100):
		t1 = threading.Thread(target=printName, args=(i,))
		t2 = threading.Thread(target=printName, args=(i+100,))
		t1.start()
		t2.start()
