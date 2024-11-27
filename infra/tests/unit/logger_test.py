from infra.logger import DataLogger



def test_log_writing(tmp_path):
    file = tmp_path / 'foo.jsonl'
    id = 'test_id'
    logger = DataLogger(file, id)
