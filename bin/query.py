#!/usr/bin/env python
"""Pulls in data stream from KPB and converts it to JSON to print to stdout

Usage:
    query.py <host> <port> <db> <user> [--limit=<N>] [--fetch_size=<N>] [--dump_file=<F>]

Options:
    --limit==<N>    How many rows to fetch in total. Set to 0 to pull in all rows. [default: 10]
    --fetch_size=<N> How many rows to fetch at a time. Results will be written out as a json list. [default: 10]
    --dump_file=<F> Where to dump the progress. Set to '' to not dump anything. [default: ]

"""
import psycopg2
import json
from time import time

if __name__ == '__main__':
    from docopt import docopt
    opt = docopt(__doc__)
    try:
        conn = psycopg2.connect("dbname='{}' user='{}' host='{}' port={}".format(opt['<db>'], opt['<user>'], opt['<host>'], opt['<port>']))
    except Exception as e:
        print('cannot connect to database')
        raise e

    cur = conn.cursor(name='tacred')
    query = """select words, subject_id, object_id, subject_ner, object_ner, subject_begin, subject_end, object_begin, object_end from test_data where subject_entity <> object_entity"""
    limit = int(opt['--limit'])
    fetch_size = int(opt['--fetch_size'])
    if limit > 0:
        query += ' limit {}'.format(limit)
    cur.execute(query)

    if opt['--dump_file']:
        prog_file = open(opt['--dump_file'], 'wb')
    else:
        prog_file = None

    num_fetched = 0
    while True:
        start = time()
        rows = cur.fetchmany(size=fetch_size)
        if not rows:
            break
        print(json.dumps(rows))
        num_fetched += len(rows)
        if prog_file is not None:
            prog_file.seek(0)
            prog_file.write('{}\n'.format(num_fetched))
            prog_file.flush()
        if limit > 0 and num_fetched > limit:
            break
    if prog_file is not None:
        prog_file.close()
