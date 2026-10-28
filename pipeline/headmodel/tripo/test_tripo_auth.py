"""An actual-style undocumented auth code must stop before any retry."""
import unittest
from unittest.mock import Mock, patch
import tripo_api


class AuthContract(unittest.TestCase):
    def test_http_401_code_two_stops_once(self):
        self.assertEqual(tripo_api.ENDPOINT,'https://openapi.tripo3d.ai/v3')
        client=object.__new__(tripo_api.Client); client.key='test-only-sentinel'
        client.session=Mock()
        client.session.request.return_value.status_code=401
        client.session.request.return_value.json.return_value={'code':2,'message':'Invalid API key'}
        with self.assertRaises(tripo_api.TripoError) as caught:
            client.request('POST','/files')
        self.assertTrue(caught.exception.fatal_auth)
        fake=Mock(); fake.calls=[]; fake.generate.side_effect=caught.exception
        with patch.object(tripo_api,'inputs',return_value=('a.png','b.png')),patch.object(tripo_api,'Client',return_value=fake):
            with self.assertRaises(tripo_api.TripoError): tripo_api.generate_all()
        self.assertEqual(fake.generate.call_count,1)


if __name__=='__main__': unittest.main()
