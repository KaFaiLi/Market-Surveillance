from datetime import time


MARKET_CLOSE_TIMES = {
    # Europe
    'GBP': {'market': 'London Stock Exchange (LSE)', 'close_time': time(16, 30), 'timezone': 'Europe/London'},
    'EUR': {'market': 'Euronext (Paris/Amsterdam/Brussels)', 'close_time': time(17, 30), 'timezone': 'Europe/Paris'},
    'CHF': {'market': 'SIX Swiss Exchange', 'close_time': time(17, 30), 'timezone': 'Europe/Zurich'},
    'SEK': {'market': 'Nasdaq Stockholm', 'close_time': time(17, 30), 'timezone': 'Europe/Stockholm'},
    'NOK': {'market': 'Oslo Bors', 'close_time': time(16, 20), 'timezone': 'Europe/Oslo'},
    'DKK': {'market': 'Nasdaq Copenhagen', 'close_time': time(17, 0), 'timezone': 'Europe/Copenhagen'},
    'PLN': {'market': 'Warsaw Stock Exchange', 'close_time': time(17, 0), 'timezone': 'Europe/Warsaw'},
    'CZK': {'market': 'Prague Stock Exchange', 'close_time': time(17, 0), 'timezone': 'Europe/Prague'},
    'HUF': {'market': 'Budapest Stock Exchange', 'close_time': time(17, 0), 'timezone': 'Europe/Budapest'},
    'RUB': {'market': 'Moscow Exchange', 'close_time': time(18, 50), 'timezone': 'Europe/Moscow'},
    # Asia Pacific
    'JPY': {'market': 'Tokyo Stock Exchange', 'close_time': time(15, 0), 'timezone': 'Asia/Tokyo'},
    'HKD': {'market': 'Hong Kong Stock Exchange', 'close_time': time(16, 0), 'timezone': 'Asia/Hong_Kong'},
    'CNY': {'market': 'Shanghai/Shenzhen Stock Exchange', 'close_time': time(15, 0), 'timezone': 'Asia/Shanghai'},
    'CNH': {'market': 'Shanghai/Shenzhen Stock Exchange', 'close_time': time(15, 0), 'timezone': 'Asia/Shanghai'},
    'SGD': {'market': 'Singapore Exchange', 'close_time': time(17, 0), 'timezone': 'Asia/Singapore'},
    'KRW': {'market': 'Korea Exchange', 'close_time': time(15, 30), 'timezone': 'Asia/Seoul'},
    'TWD': {'market': 'Taiwan Stock Exchange', 'close_time': time(13, 30), 'timezone': 'Asia/Taipei'},
    'INR': {'market': 'National Stock Exchange of India', 'close_time': time(15, 30), 'timezone': 'Asia/Kolkata'},
    'AUD': {'market': 'Australian Securities Exchange', 'close_time': time(16, 0), 'timezone': 'Australia/Sydney'},
    'NZD': {'market': 'New Zealand Exchange', 'close_time': time(17, 0), 'timezone': 'Pacific/Auckland'},
    'MYR': {'market': 'Bursa Malaysia', 'close_time': time(17, 0), 'timezone': 'Asia/Kuala_Lumpur'},
    'THB': {'market': 'Stock Exchange of Thailand', 'close_time': time(16, 30), 'timezone': 'Asia/Bangkok'},
    'IDR': {'market': 'Indonesia Stock Exchange', 'close_time': time(15, 0), 'timezone': 'Asia/Jakarta'},
    'PHP': {'market': 'Philippine Stock Exchange', 'close_time': time(15, 30), 'timezone': 'Asia/Manila'},
    'VND': {'market': 'Ho Chi Minh Stock Exchange', 'close_time': time(15, 0), 'timezone': 'Asia/Ho_Chi_Minh'},
    # North America
    'USD': {'market': 'NYSE/NASDAQ', 'close_time': time(16, 0), 'timezone': 'America/New_York'},
    'CAD': {'market': 'Toronto Stock Exchange', 'close_time': time(16, 0), 'timezone': 'America/Toronto'},
    'MXN': {'market': 'Mexican Stock Exchange', 'close_time': time(15, 0), 'timezone': 'America/Mexico_City'},
}
