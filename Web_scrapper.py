# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:31:23 2022

@author: Bhavana
"""
from selenium.common import TimeoutException
import os
import pandas as pd
import time
import datetime

# libraries to crawl websites
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

LIMIT_HOTEL = 15
LIMIT_REVIEWS_PER_HOTEL = 100

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# %%%%
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 5)
pd.set_option('display.max_colwidth', 10)
pd.set_option('display.width', 800)
path = "."
os.chdir(path)

# %%

SEED_URL = "https://www.booking.com/searchresults.html?aid=304142&label=gen173nr%20%5C%20" \
           "-1FCAEoggI46AdIM1gEaGyIAQGYAQm4AQfIAQzYAQHoAQH4AQuIAgGoAgO4AtPev5wGwAIB0g%20%5C" \
           "%20IkMzczODQ3OWMtN2M5Ny00NjA5LThjYWQtYzM0YTBlOGQzN2Y12AIG4AIB&sid=b0606c9325481785d6f3a8af5a0a6aa3&tmpl" \
           "=searchresults&checkin_month=12&checkin_monthday=7&checkin_year=2022&checkout_month=12&checkout_monthday" \
           "=8&class_interval=1&dest_type=city&dtdisc=0&efdco=1&from_sf=1&group_children=0&inac=0&index_postcard=0" \
           "&label_click=undef&no_rooms=1&offset=0&postcard=0&raw_dest_type=city&room1=A%2CA&sb_price_type=total" \
           "&shw_aparth=1&slp_r_match=0&src=index&src_elem=sb&srpvid=f4055e706d01020b&ss=Boston&ss_all=0&ssb=empty" \
           "&sshis=0&ssne=Boston&ssne_untouched=Boston&changed_currency=1&selected_currency=USD&top_currency=1 "


# Creating the list of links.
# Collect urls of all hotels
def crawl_boston_hotel(seed_url, hotel_limit):
    driver.get(seed_url)
    time.sleep(3)
    links_set_to_avoid_duplicate_entry = {}
    hotel_link_data = []
    while True:
        links_to_hotel = driver.find_elements(By.XPATH, "//div[@class='dd023375f5']")
        counter = 0
        while counter in range(len(links_to_hotel)):
            hotel = {'scrapping_date': datetime.datetime.now()}  # initializing object
            inner_html = BeautifulSoup(links_to_hotel[counter].get_attribute('innerHTML'))
            try:
                link = inner_html.find('a', attrs={'class': 'e13098a59f'})['href']
            except Exception as ex:
                link = ""
                if link in links_set_to_avoid_duplicate_entry:
                    continue
            hotel['link'] = link
            links_set_to_avoid_duplicate_entry = link
            try:
                name = inner_html.find('div', attrs={'class': 'fcab3ed991 a23c043802'}).text
            except Exception as ex:
                name = ""
            hotel['name_of_hotel'] = name
            hotel_link_data.append(hotel)
            counter = counter + 1
            if len(hotel_link_data) >= hotel_limit:
                return hotel_link_data
        try:
            WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Next page']"))).click()
        except TimeoutException as tm:
            break
        time.sleep(5)


df = pd.DataFrame(crawl_boston_hotel(SEED_URL, LIMIT_HOTEL))
# printing the dataframe
print(df.shape)
df.to_csv("hotel_link_data.csv", index=False)


def crawl_hotel_reviews(hotel_links, review_limit_per_hotel):
    hotel_links = hotel_links.reset_index()  # make sure indexes pair with number of rows
    # now scrape hotel details and reviews.
    hotel_reviews = []
    for index, row in hotel_links.iterrows():
        url = row['link']
        driver.get(url)
        time.sleep(1)
        hotel_details = driver.find_elements(By.XPATH, "//*[@id='basiclayout']/div[1]/div[1]/div/div[2]")
        num = 0
        while num in range(len(hotel_details)):
            soup = BeautifulSoup(hotel_details[num].get_attribute('innerHTML'))
            review_count_per_hotel = 0
            try:
                hotel_name = soup.find('h2', attrs={'class': 'd2fee87262 pp-header__title'}).text
                address = soup.find('span',
                                    attrs={'class': 'hp_address_subtitle js-hp_address_subtitle jq_tooltip'}).text
                review_score = soup.find('div', attrs={'class': 'b5cd09854e d10a6220b4'}).text
                review_rating_category = soup.find('div', attrs={'class': 'b5cd09854e f0d4d6a2f5 e46e88563a'}).text
                customer_review_count = soup.find('div', attrs={'class': 'd8eab2cf7f c90c0a70d3 db63693c62'}).text
            except Exception as ex:
                print("Exception occurred while getting hotel details skipping this record :", ex)
                num = num + 1
                continue
            try:
                WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='read-all-actionable']"))).click()
            except TimeoutException as tm:
                break
            while True:
                reviews = driver.find_elements(By.XPATH, "//li[@class='review_list_new_item_block']")
                k = 0
                while k in range(len(reviews)):
                    hotel_reviews_record = {'scrapping_date': datetime.datetime.now(),
                                            'hotel_name': hotel_name,
                                            'hotel_address': address,
                                            'review_score': review_score,
                                            'review_rating_category': review_rating_category,
                                            'customer_review_count': customer_review_count}  # initializing object
                    review_soup = BeautifulSoup(reviews[k].get_attribute('innerHTML'))
                    try:
                        hotel_reviews_record['room_type_block'] = review_soup.find('li',
                                                                                   attrs={'class': 'bui-list__item '
                                                                                                   'review-block__room-info--disabled'}). \
                            findChild('div', attrs={'class': 'bui-list__body'}).text
                        hotel_reviews_record['num_of_days_stayed'] = review_soup.find('ul', attrs={
                            'class': 'bui-list bui-list--text '
                                     'bui-list--icon bui_font_caption '
                                     'c-review-block__row '
                                     'c-review-block__stay-date'}). \
                            findChild('div', attrs={'class': 'bui-list__body'}).text
                        hotel_reviews_record['date_of_stay'] = review_soup.find('ul', attrs={
                            'class': 'bui-list bui-list--text bui-list--icon '
                                     'bui_font_caption c-review-block__row '
                                     'c-review-block__stay-date'}). \
                            findChild('span', attrs={'class': 'c-review-block__date'}).text
                        hotel_reviews_record['traveller_type'] = review_soup.find('ul', attrs={
                            'class': 'bui-list bui-list--text bui-list--icon '
                                     'bui_font_caption '
                                     'review-panel-wide__traveller_type '
                                     'c-review-block__row'}). \
                            findChild('div', attrs={'class': 'bui-list__body'}).text
                        hotel_reviews_record['review_score_badge'] = review_soup.find('div',
                                                                                      attrs={
                                                                                          'class': 'bui-review-score__badge'}).text
                        # hotel_reviews_record['review_title'] = review_soup.find('h3', attrs={
                        #     'class': 'c-review-block__title c-review__title--ltr '
                        #              ' '}).text
                        review_text = review_soup.findChildren('span',
                                                               attrs={'class': 'c-review__body'})
                        if len(review_text) == 0:
                            continue
                        hotel_reviews_record['review_text_positive'] = review_text[0].text
                        if len(review_text) > 1:
                            hotel_reviews_record['review_text_negative'] = review_text[1].text
                        else:
                            hotel_reviews_record['review_text_negative'] = 'N/A'
                        hotel_reviews_record['review_date'] = review_soup.find('span', attrs={
                            'class': 'c-review-block__date'}).text
                    except Exception as ex:
                        print("Exception occurred while getting hotel reviews, skipping this record :", ex)
                        k = k + 1
                        continue
                    hotel_reviews.append(hotel_reviews_record)
                    review_count_per_hotel += 1
                    if review_count_per_hotel >= review_limit_per_hotel:
                        break
                    k = k + 1
                if review_count_per_hotel >= review_limit_per_hotel:
                    num += 1
                    break
                try:
                    WebDriverWait(driver, 20).until(
                        EC.element_to_be_clickable((By.XPATH, "//a[@class='pagenext']"))).click()
                except TimeoutException as tm:
                    break
                time.sleep(2)
            num += 1
    return hotel_reviews


hotel_link_df = pd.read_csv("hotel_link_data.csv")
hotel_review_list = crawl_hotel_reviews(hotel_link_df, LIMIT_REVIEWS_PER_HOTEL)
print(hotel_review_list)
final_df = pd.DataFrame(hotel_review_list)
print(final_df.shape)
final_df.to_csv("hotel_reviews.csv", index=False)
