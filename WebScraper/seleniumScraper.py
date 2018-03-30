import itertools
import string
import sys
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from StudentModel import Student



path_to_chromedriver = './chromedriver2.37' # change path as needed


credentials = []

timeout =5
driver = None
main_window = None
foot_hold_windows = []



def getCredentials(): 
	file = open("../credentials.txt", "r")
	for line in file: 
		credentials.append(line)


def setUpDriver():
	chrome_options = Options()
	chrome_options.add_argument("--ignore-certificate-errors")
	chrome_options.add_argument("start-maximized")
	chrome_options.add_argument("disable-infobars")
	chrome_options.add_argument("--disable-extensions")
	chrome_options.add_argument("--test-type")
	global driver
	driver = webdriver.Chrome(executable_path = path_to_chromedriver, chrome_options=chrome_options)



def goBack(): 
	 driver.execute_script("window.history.go(-1)")

def openElementInNewWindow(): 
	Browser.execute_script('document.getElementById("someID")').setAttribute("target","_blank")


def openNewWindow(): 
	body = driver.find_element_by_tag_name('body')
	body.send_keys(Keys.COMMAND + 'n')
	switchToWindow(1)

def switchToWindow(window=0):
	driver.switch_to_window(driver.window_handles[window])
	# driver.get("http://facebook.com")
	# print "New window ", driver.title

def getCurrentWindow(): 
	return driver.current_window_handle

def close_window(): 
	# driver.find_element_by_tag_name('body').send_keys(Keys.COMMAND + 'w')
	driver.close()


def writePersonToFile(name, major, email): 
	file = open("../directory.txt", "a")
	# file.write("{} \n {} \n {} \n {} \n {} \n\n".format(person.name, person.major, person.classification, person.phone, person.email))
	file.write("{} \n {} \n {} \n\n".format(name, major, email))
	# file.write(email + "\n")
	# file.write(person)
	file.close()




def copyStudentInfo(): 
	person = Student()
	personItems = []
	personItems = driver.find_elements_by_class_name('dir-Person-item')

	# for item in personItems: 
	# 	if(person.major == None): 
	# 		person.major = item.text 
	# 	elif(person.classification == None):
	# 		person.classification = item.text
	# 	elif(person.phone == None): 
	# 		person.phone = item.text
	# 	elif(person.email ==None): 
	# 		person.email = item.text
	name = driver.find_element_by_tag_name('h1').text
	
	email = None
	major = None

	if(len(personItems) >= 3):
		major = personItems[0].text 
		email = personItems[-2].text
	elif(len(personItems) >=2):
		major = personItems[0]
		email = personItems[-2].text
	elif(len(personItems) >=1): 
		major = personItems[0].text

	writePersonToFile(name, major, email)


			

def signInToShibboleth(): 
	time.sleep(2)
	if(getId('okta-signin-username')):
		driver.find_element_by_xpath('//*[@id="okta-signin-username"]').send_keys(credentials[0])
		# for input in inputs: 
	if(getId('okta-signin-password')):	
		driver.find_element_by_id('okta-signin-password').send_keys(credentials[1])
	if(getId('okta-signin-submit')):
		driver.find_element_by_id('okta-signin-submit').click()




def getCombination(length=2): 
	result = []
	for guess in itertools.product(): 
		for num_length in range(0, length): 
			for num_length in itertools.combinations_with_replacement(string.digits, num_length): 
				guess = ''.join(guess)
				result.append(guess)


def navigate_back_to_search():
	driver.find_element_by_xpath('//*[@id="content"]/div/p/a').click()
	if(getId('individuals')):
			driver.find_element_by_xpath('//*[@id="individual_type__students"]').click()

		


def search_letter_combo(letters):
	driver.find_element_by_id('individuals').clear()
	driver.find_element_by_id('individuals').send_keys(letters)
	driver.find_element_by_id('individual_type__students').click()
	driver.find_element_by_id('search_individuals').click()



def get_people_listings(): 
	ulList = driver.find_element_by_class_name('dir-Listing')
	items = ulList.find_elements_by_tag_name('li')
	return items

def get_page_listings(): 
	pages = driver.find_element_by_class_name('wd-Pagination')
	pageList = pages.find_elements_by_tag_name('li')
	if len(pageList) > 11 : 
		del pageList[11:len(pageList)]
	del pageList[0]
	return pageList

def log_me_in(item): 
	item.click()
	if(getClass('right-buttons')):
		# if(getClassBy)
		if(getXPath('//*[@id="content"]/div/div[1]/a')):
			driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/a').click()
			signInToShibboleth()
			time.sleep(2)
			# Click the new search button
			navigate_back_to_search()
		else:
			goBack()
	else:
		goBack()

def login(): 
	search_letter_combo("aa")
	items = get_people_listings()
	log_me_in(items[0])
	
def open_all_pages(): 
	pageList = get_page_listings()

	for page in pageList: 	
		anchor = page.find_element_by_tag_name('a')
		open_in_new_page(anchor)
	return len(pageList)

def loop_through_pages(pageCount):
	global main_window
	global foot_hold_windows
	foot_hold_windows = driver.window_handles

	# for handle in driver.window_handles[1:len(driver.window_handles)]:
	for handle in foot_hold_windows:
		if(handle != main_window):
			driver.switch_to_window(handle)
			# foot_hold_window = len(driver.window_handles)-1
			get_people_info()
			# close_window()
			# foot_hold_windows.remove(handle)
			

	# get_people_info()
	# driver.switch_to_window(main_window)
	# pageList = get_page_listings()
	# for page in pageList: 	
	# 	anchor = page.find_element_by_tag_name('a')
	# 	open_in_new_page(anchor)
	# 	close_window()
	# 	main_window = driver.current_window_handle
	# 	get_people_info()
		# switch_window_get_student_info_and_close()
	

def switch_window_get_student_info_and_close():
	# driver.switch_to_window(driver.window_handles[len(driver.window_handles)-1])
	for handle in driver.window_handles:
		if(handle != main_window):
			driver.switch_to_window(handle)
			get_people_info()
			close_window()

def open_in_new_page(element):
	element.send_keys(Keys.COMMAND + Keys.RETURN)

def move_to_right_one_tab(): 
	driver.find_element_by_tag_name('body').send_keys(Keys.COMMAND + Keys.NUMPAD2)

def move_to_left_one_tab(): 
	driver.find_element_by_tag_name('body').send_keys(Keys.COMMAND + Keys.SHIFT+ Keys.NUMPAD2)

def get_student_info_from_open_pages(count):
	for handle in driver.window_handles:
		if handle not in foot_hold_windows:
			driver.switch_to_window(handle)
			copyStudentInfo()
			# time.sleep(1)
			close_window()
		# count -= 1
	# else:
		# break;



def get_people_info():
	items = get_people_listings()
	for item in items: 
		# item.click()
		open_in_new_page(item.find_element_by_tag_name("a"))
		# move_to_right_one_tab()
		# time.sleep(3)
		# copyStudentInfo()
		# close_window()
		# Go back to the page with all the directory listings so we can keep using these items
		# goBack()
	get_student_info_from_open_pages(len(items))



def remove_all_foot_hold_elements_that_are_not_main_window(): 
	global foot_hold_windows
	print len(foot_hold_windows)
	for window in foot_hold_windows:
		if(window != main_window): 
			driver.switch_to_window(window)
			close_window()

def main(): 
	getCredentials()
	setUpDriver()
	driver.get(credentials[2])
	login()

	# all possible two letter character combinations
	characters = [''.join(i) for i in itertools.product(string.ascii_lowercase, repeat = 2)]

	global main_window
	main_window = driver.current_window_handle
	# driver.switch_to_frame('mainFrame')
	for combo in characters: 
		driver.switch_to_window(main_window)
		search_letter_combo(combo)

		# First Get people listings on all other pages
		pageCount = open_all_pages() 
		loop_through_pages(pageCount)

		# Then come back and get listing for main page
		global foot_hold_windows
		remove_all_foot_hold_elements_that_are_not_main_window()
		foot_hold_windows = []
		foot_hold_windows.append(main_window)
		driver.switch_to_window(main_window)
		get_people_info()


		
		
def getXPath(path):
	try:
	    element_present = EC.presence_of_element_located((By.XPATH, path))
	    WebDriverWait(driver, timeout).until(element_present)
	except TimeoutException:
	    print "Timed out waiting for page to load"
	else: 
		return True
		
def getLinkText(link_text):
	try:
	    element_present = EC.presence_of_element_located((By.LINK_TEXT, link_text))
	    WebDriverWait(driver, timeout).until(element_present)
	except TimeoutException:
	    print "Timed out waiting for page to load"
	else: 
		return True

def getName(element_name):
	try:
	    element_present = EC.presence_of_element_located((By.NAME, element_name))
	    WebDriverWait(driver, timeout).until(element_present)
	    return True
	except TimeoutException:
	    print "Timed out waiting for page to load"

def getId(element_id): 
	try:
	    element_present = EC.presence_of_element_located((By.ID, element_id))
	    WebDriverWait(driver, timeout).until(element_present)
	except TimeoutException:
	    print "Timed out waiting for page to load"
	else: 
		return True

def getClass(element_class):
	try:
	    element_present = EC.presence_of_element_located((By.CLASS_NAME, element_class))
	except TimeoutException:
	    print "Timed out waiting for page to load"
	else:
		return True

if __name__ == "__main__": 
	main(); 