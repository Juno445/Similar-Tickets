import builtins
import io
import os
from unittest import mock
import pandas as pd

os.environ.setdefault('API_KEY', 'x')
os.environ.setdefault('DOMAIN', 'example')

from src.main import sanitize_html, load_and_prepare_data, fetch_dispatch_tickets


def test_sanitize_html():
    html = '<p>Hello <b>World</b></p>'
    assert sanitize_html(html) == 'HelloWorld'


def test_load_and_prepare_data(tmp_path):
    data = ('Ticket ID,Subject,Requester Email,Date Created,Description\n'
            '1,Test,a@b.com,2024-01-01,Desc')
    file = tmp_path / 'tickets.csv'
    file.write_text(data)
    df = load_and_prepare_data(file)
    assert len(df) == 1
    assert list(df['Ticket ID'])[0] == 1


def test_fetch_dispatch_tickets(tmp_path):
    fake_response = {
        'tickets': [
            {'id': 1, 'subject': 'Hello', 'requester_id': 'a@b.com', 'created_at': '2024-01-01', 'description': '<p>desc</p>'}
        ]
    }
    empty_response = {'tickets': []}
    with mock.patch('requests.get') as mock_get:
        mock_get.side_effect = [
            mock.Mock(status_code=200, json=lambda: fake_response),
            mock.Mock(status_code=200, json=lambda: empty_response)
        ]
        tickets = fetch_dispatch_tickets(tmp_path / 'out.csv')

    assert len(tickets) == 1
    assert tickets[0]['Subject'] == 'Hello'
