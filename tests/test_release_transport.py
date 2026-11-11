"""Offline checks of authenticated release URL resolution and atomic downloads."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error
import urllib.request

from scripts import release_transport as rt


class Response(io.BytesIO):
    def __init__(self, body=b'archive', headers=None):
        super().__init__(body)
        self.headers = headers or {'Content-Length': str(len(body))}


class Transport(unittest.TestCase):
    def setUp(self):
        rt._asset_ids.cache_clear()

    def test_private_release_uses_asset_api(self):
        with mock.patch.object(rt, '_json_get', side_effect=[{'id': 7}, [{'id': 42, 'name': 'x.tar.gz'}]]) as get:
            self.assertEqual(rt.download_url('https://github.com/VKHS/CR-9/releases/download/tag/x.tar.gz', 'secret'),
                             'https://api.github.com/repos/VKHS/CR-9/releases/assets/42')
            self.assertEqual(get.call_count, 2)
            self.assertIn('releases/7/assets', get.call_args.args[0])

    def test_metadata_paginated(self):
        first = [{'id': i + 1, 'name': f'part{i}'} for i in range(100)]
        with mock.patch.object(rt, '_json_get', side_effect=[{'id': 9}, first, [{'id': 500, 'name': 'last'}]]) as get:
            url = rt.download_url('https://github.com/o/r/releases/download/v1/last', 'secret')
            self.assertTrue(url.endswith('/500'))
            self.assertIn('page=2', get.call_args.args[0])

    def test_public_download_does_not_need_metadata(self):
        url = 'https://github.com/o/r/releases/download/v1/a'
        with mock.patch.object(rt, '_json_get') as get:
            self.assertEqual(rt.download_url(url, None), url)
            get.assert_not_called()

    def test_mirror_never_receives_github_token(self):
        for url in ('https://mirror.example/a', 'http://127.0.0.1:1234/a', 'https://github.com/a'):
            self.assertNotIn('Authorization', rt._headers(url, 'secret', binary=True))
        self.assertEqual(rt._headers('https://api.github.com/repos/o/r', 'secret', binary=True)['Authorization'],
                         'Bearer secret')

    def test_cross_origin_redirect_strips_credentials(self):
        request = urllib.request.Request('https://api.github.com/repos/o/r/releases/assets/1',
                                         headers={'Authorization': 'Bearer secret', 'Cookie': 'session'})
        out = rt.CredentialSafeRedirect().redirect_request(request, None, 302, 'Found', {}, 'https://release-assets.githubusercontent.com/a')
        self.assertNotIn('authorization', {k.lower() for k in out.headers})
        self.assertNotIn('cookie', {k.lower() for k in out.headers})

    def test_same_origin_redirect_keeps_authentication(self):
        request = urllib.request.Request('https://api.github.com/a', headers={'Authorization': 'Bearer secret'})
        out = rt.CredentialSafeRedirect().redirect_request(request, None, 302, 'Found', {}, '/b')
        self.assertEqual(out.get_header('Authorization'), 'Bearer secret')

    def test_https_downgrade_refused(self):
        request = urllib.request.Request('https://api.github.com/a')
        with self.assertRaises(urllib.error.URLError):
            rt.CredentialSafeRedirect().redirect_request(request, None, 302, 'Found', {}, 'http://api.github.com/b')

    def test_missing_asset_is_not_silently_substituted(self):
        with mock.patch.object(rt, '_json_get', side_effect=[{'id': 7}, []]):
            with self.assertRaises(FileNotFoundError):
                rt.download_url('https://github.com/o/r/releases/download/v1/absent', 'secret')

    def test_duplicate_asset_refused(self):
        with mock.patch.object(rt, '_json_get', side_effect=[{'id': 7}, [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'a'}]]):
            with self.assertRaises(ValueError):
                rt.download_url('https://github.com/o/r/releases/download/v1/a', 'secret')

    def test_bad_urls_rejected(self):
        for url in ('file:///etc/passwd', 'https://user:password@github.com/a', 'https://api.github.com:wrong/a'):
            with self.assertRaises(ValueError):
                rt._headers(url, 'secret', binary=True)

    def test_encoded_tag_supported(self):
        self.assertEqual(rt.release_coordinates('https://github.com/o/r/releases/download/group%2Fv1/a%20b'),
                         ('o', 'r', 'group/v1', 'a b'))

    def test_atomic_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data'
            path.write_bytes(b'old')
            opener = mock.Mock()
            opener.open.return_value = Response(b'new')
            with mock.patch.object(rt.urllib.request, 'build_opener', return_value=opener):
                rt.download('https://mirror.example/data', path, token='secret')
            self.assertEqual(path.read_bytes(), b'new')
            request = opener.open.call_args.args[0]
            self.assertIsNone(request.get_header('Authorization'))
            self.assertEqual(list(Path(folder).glob('*.download')), [])

    def test_truncation_preserves_original_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data'
            path.write_bytes(b'old')
            opener = mock.Mock()
            opener.open.return_value = Response(b'new', {'Content-Length': '100'})
            with mock.patch.object(rt.urllib.request, 'build_opener', return_value=opener), self.assertRaises(OSError):
                rt.download('https://mirror.example/data', path)
            self.assertEqual(path.read_bytes(), b'old')
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ['data'])

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            real = Path(folder) / 'real'
            real.write_bytes(b'untouched')
            link = Path(folder) / 'data'
            link.symlink_to(real)
            with self.assertRaises(ValueError):
                rt.download('https://mirror.example/a', link)
            self.assertEqual(real.read_bytes(), b'untouched')


if __name__ == '__main__':
    unittest.main()
