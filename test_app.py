import main_api
from fastapi.testclient import TestClient
import json

client = TestClient(main_api.app)

def test_response():
 '''
    check server accessibility
 '''
  response = client.get('/')
  assert response.status_code == 200
  assert response.json() == {'message': "Greats! It's work!"}

def test_message():
  '''
    check message handler
  '''
  response = client.post(
      '/check/message/',
       json={'text':'ты дурак', 'mode':'all'})
  assert response.status_code == 200

def test_dection_response():
  '''
    check message body structure
  '''
  js_object ={'text':'ты дурак', 'mode':'all'}
  response = client.post(
      '/check/message/',
       json=js_object)
  res_object = response.json()
  assert response.status_code == 200
  assert 'text' in res_object
  assert 'neutral' in res_object
  assert 'toxic' in res_object
  assert js_object['text'] == res_object['text']


def test_msgs_processing():
  '''
    check the completeness response on posted messages
  '''
  N_OBJECTS=5
  js_object ={'text':'ты дурак', 'mode':'all'}
  js_objects = [js_object]*N_OBJECTS
  response = client.post(
      '/check/messages/',
       json=js_objects)
  messages = json.loads(response.text)
  assert response.status_code == 200
  assert len(messages) == N_OBJECTS





