import argparse
import json
from utils.constants import BASE_URL
from utils.requester import Requester

def get_arguments():
    parser = argparse.ArgumentParser(usage='Scraps all wine data from Vivino.')
    parser.add_argument('output_file', help='Output .json file', type=str)
    parser.add_argument('-start_page', help='Starting page identifier', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    output_file = args.output_file
    start_page = args.start_page
    r = Requester(BASE_URL)

    payload = {
        "price_range_min": 10,
        "price_range_max": 2000,
    }

    res = r.get('explore/explore?', params=payload)
    n_matches = res.json()['explore_vintage']['records_matched']

    print(f'Number of matches: {n_matches}')

    # Create a list to hold all the data
    all_wines = []

    page_size = 25  # You may need to adjust this based on API limitations
    total_pages = max(1, int(n_matches / page_size)) + 1

    for i in range(start_page, total_pages + 1):
        payload['page'] = i
        payload['page_size'] = page_size

        print(f'Page: {payload["page"]}')

        res = r.get('explore/explore', params=payload)
        matches = res.json()['explore_vintage']['matches']

        for match in matches:
            wine = match['vintage']['wine']
            price = match['price']
            wine['price'] = price

            print(f'Scraping data from wine: {wine["name"]}')

            wine_data = {
                'name': wine['name'],
                'price': wine['price'],
                'taste': r.get(f'wines/{wine["id"]}/tastes').json()['tastes'],
                'reviews': r.get(f'wines/{wine["id"]}/reviews').json()['reviews']
            }

            all_wines.append(wine_data)

    # Open the output .json file
    with open(output_file, 'w') as f:
        # Dump the data
        json.dump({"wines": all_wines}, f)
