import pytest
from fastapi.testclient import TestClient
from fea.server import app
from fea.valuation import evaluate_inventory

def item(**changes):
    return dict(name='제품 A',quantity='100',cost='10000',price='9500',completion='500',selling='300',**changes) if not changes else {**item(),**changes}

def test_item_losses_are_not_offset_by_other_item_gains():
    r=evaluate_inventory([item(),item(price='12000')])
    assert r['loss']=='130000'
    assert r['carrying']=='1870000'
    assert r['rows'][0]['nrv']=='8700'

def test_decimal_calculation_is_exact():
    r=evaluate_inventory([item(quantity='0.10',cost='0.30',price='0.20',completion='0',selling='0')])
    assert r['loss']=='0.0100'

@pytest.mark.parametrize('change',[{'quantity':'NaN'},{'cost':'-1'},{'selling':'Infinity'},{'price':'0'},{'cost':'1.001'}])
def test_invalid_or_unsupported_conditions(change):
    with pytest.raises(ValueError):evaluate_inventory([item(**change)])

def test_valuation_without_dart_or_llm_key():
    with TestClient(app) as client:
        r=client.post('/api/valuation',json={'items':[item()]})
        assert r.status_code==200
        assert r.json()['loss']=='130000'
        assert client.post('/api/valuation',json={'items':[]}).status_code==422
