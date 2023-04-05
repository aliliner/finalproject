import scrapy
import re
from bs4 import BeautifulSoup

def get_arr_of_content_1(content):
    content_a = content.find_all('a')
    content_arr = []

    if len(content_a) > 0:
        for iter_a in content_a:
            content_arr.append(iter_a.get_text())

    else:
        content_arr.append(content.get_text())
    return content_arr


def get_arr_of_content_2(content):
    content_a = content.find_all('a')
    content_arr = []

    if len(content_a) > 0:
        for iter_a in content_a:
            content_arr.append(clear_value(iter_a.get_text()))

    else:
        content_arr.append(clear_value(content.get_text()))
    return content_arr


def clear_value(elem):
    return elem.replace('\n', '', 1).strip()



class WineSpider(scrapy.Spider):
    name        = 'smpl'
    page_number = 2
    start_urls  = ['https://simplewine.ru/catalog/vino/']

    def parse(self, response):
        for link in response.css('a.product-snippet__image-href::attr(href)'):
            yield response.follow(link, callback=self.parse_wine)

        next_page = 'https://simplewine.ru/catalog/vino/page' + str(WineSpider.page_number) + '/'

        if WineSpider.page_number <= 186:
            WineSpider.page_number += 1
            yield response.follow(next_page, callback = self.parse)


    def parse_wine(self, response):
        dict_attr = {}

        price_prod   = response.css('div.product-buy__price::text').get()
        raiting_prod = response.css('div.product-info__raiting-count::text').get()

        dict_attr.update({'price' : price_prod})
        dict_attr.update({'rating': raiting_prod})


        content1_resp = response.css('.product-info__list').get()
        content1_soup = BeautifulSoup(content1_resp, 'html.parser')
        content1      = content1_soup.find_all('div', {'class': 'product-info__list-item'})

        for elem in content1:
            content_par = elem.find('div', {'class': 'product-info__list-type'})
            content_val = elem.find('div', {'class': 'product-info__list-desc'})

            title = get_arr_of_content_1(content_par)
            value = get_arr_of_content_1(content_val)

            title_str = '##'.join(title)
            value_str = '##'.join(value)

            dict_attr.update({title_str: value_str})


        content2_resp = response.css('.characteristics-params__list').get()
        content2_soup = BeautifulSoup(content2_resp, 'html.parser')
        content2      = content2_soup.find_all('div', {'class': 'characteristics-params__item'})

        for elem in content2:
            content_par = elem.find('dt', {'class': 'characteristics-params__title'})
            content_val = elem.find('dd', {'class': 'characteristics-params__value'})

            title = get_arr_of_content_2(content_par)
            value = get_arr_of_content_2(content_val)

            title_str = '##'.join(title)
            value_str = '##'.join(value)

            dict_attr.update({title_str: value_str})

        article_prod = response.css('a.product__header-fav.product-card__favorite::attr(data-id)').get()
        article_altr = response.css('div.product-card-type-a__header-info span::text').get()
        article_altr = re.findall(r'(.*)( *: *)([0-9]*)', article_altr)[0][2]


        for attr in dict_attr:
            yield {
                'code'      : article_prod,
                'code_check': article_altr,
                'title'     : attr,
                'value'     : dict_attr[attr]
            }























