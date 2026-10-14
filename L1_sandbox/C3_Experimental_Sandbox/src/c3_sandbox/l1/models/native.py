"""Provider protocols stay native; only calls and usage get normalized projections."""
from __future__ import annotations
import base64
import copy
import json
from urllib.parse import quote
from ..core.contracts import Call, Reply, ProtocolError, ConfigurationError, Result
from .http import NativeHTTP


def parse_json(body: bytes):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d: raise ProtocolError('duplicate JSON key')
            d[k]=v
        return d
    try:
        x=json.loads(body,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
        if not isinstance(x,dict): raise ValueError('object required')
        return x
    except (ValueError,UnicodeError,TypeError) as exc:
        raise ProtocolError('Invalid native JSON') from exc


def _arguments(value):
    if not isinstance(value,dict): raise ProtocolError('Tool arguments must be JSON object')
    return value

class GeminiBackend:
    def __init__(self, config):
        self.config=config; self.http=NativeHTTP(config)
    def identity(self):
        return {'adapter':'gemini','kind':self.config['kind'],'provider':'google',
                'requested_model':self.config['model'],'reported_revision':self.config.get('reported_revision'),
                'tokenizer_digest':self.config.get('tokenizer_digest'),'native_protocol':'generateContent',
                'weight_digest':None}
    def initial(self,system,prompt):
        return {'systemInstruction':{'parts':[{'text':system}]},
                'contents':[{'role':'user','parts':[{'text':prompt}]}]}
    def request(self,conversation,tools):
        body=copy.deepcopy(conversation)
        body['tools']=[{'functionDeclarations':copy.deepcopy(tools)}]
        body['generationConfig']=copy.deepcopy(self.config['parameters'])
        if body['generationConfig'].get('candidateCount',1)!=1: raise ConfigurationError('candidateCount must be 1')
        return body
    def send(self,body,timeout):
        model=self.config['model'].removeprefix('models/')
        return self.http.post(f'{self.http.base}/models/{quote(model,safe="")}:generateContent',body,timeout,gemini=True)
    def parse(self,body,request_id):
        value=parse_json(body)
        if 'error' in value: raise ProtocolError('Provider returned error envelope')
        candidates=value.get('candidates')
        if not isinstance(candidates,list) or len(candidates)!=1: raise ProtocolError('Expected exactly one Gemini candidate')
        c=candidates[0]; content=c.get('content')
        if not isinstance(content,dict) or content.get('role','model')!='model' or not isinstance(content.get('parts'),list):
            raise ProtocolError('Missing native Gemini content')
        calls=[]; texts=[]; seen=set()
        for index,part in enumerate(content['parts']):
            if not isinstance(part,dict): raise ProtocolError('Malformed Gemini part')
            if 'functionCall' in part:
                fc=part['functionCall']
                if not isinstance(fc,dict) or not isinstance(fc.get('name'),str): raise ProtocolError('Invalid functionCall')
                native_id=fc.get('id')
                if native_id is not None:
                    if not isinstance(native_id,str) or not native_id or native_id in seen: raise ProtocolError('Invalid or duplicate call ID')
                    seen.add(native_id)
                calls.append(Call(f'{request_id}:c{index}',fc['name'],_arguments(fc.get('args',{})),native_id))
            elif isinstance(part.get('text'),str) and not part.get('thought'):
                texts.append(part['text'])
        if c.get('finishReason') not in (None,'STOP'):
            raise ProtocolError('Non-complete Gemini candidate: '+str(c.get('finishReason')))
        return Reply(tuple(calls),copy.deepcopy(content),value.get('usageMetadata'),value.get('modelVersion'),
                     c.get('finishReason'),'\n'.join(texts) or None)
    def accept(self,conversation,reply,results):
        new=copy.deepcopy(conversation)
        # Retain every native part and thoughtSignature without reconstruction.
        new['contents'].append(copy.deepcopy(reply.native_message))
        if results:
            parts=[]
            for call,result in results:
                fc={'name':call.name,'response':copy.deepcopy(result.observation)}
                if call.native_id: fc['id']=call.native_id
                parts.append({'functionResponse':fc})
                if result.image:
                    parts.append({'inlineData':{'mimeType':result.image_mime,
                        'data':base64.b64encode(result.image).decode('ascii')}})
            new['contents'].append({'role':'user','parts':parts})
        return new

class OpenAICompatibleBackend:
    """API or locally served weights; not an in-process MLX dependency."""
    def __init__(self,config):
        self.config=config; self.http=NativeHTTP(config)
        forbidden={'model','messages','tools','stream','n','tool_choice','parallel_tool_calls'}
        if forbidden.intersection(config['parameters']): raise ConfigurationError('parameters override protocol-reserved fields')
    def identity(self):
        return {'adapter':'openai-compatible','kind':self.config['kind'],
                'provider':self.config.get('provider','openai-compatible'),
                'requested_model':self.config['model'],'reported_revision':self.config.get('reported_revision'),
                'tokenizer_digest':self.config.get('tokenizer_digest'),'native_protocol':'chat-completions',
                'weight_digest':None}
    def initial(self,system,prompt):
        return {'messages':[{'role':'system','content':system},{'role':'user','content':prompt}]}
    def request(self,conversation,tools):
        return {'model':self.config['model'],'messages':copy.deepcopy(conversation['messages']),
                'tools':[{'type':'function','function':copy.deepcopy(t)} for t in tools],
                'tool_choice':'auto','stream':False,'n':1,**copy.deepcopy(self.config['parameters'])}
    def send(self,body,timeout):
        return self.http.post(self.http.base+'/chat/completions',body,timeout)
    def parse(self,body,request_id):
        value=parse_json(body)
        choices=value.get('choices')
        if 'error' in value or not isinstance(choices,list) or len(choices)!=1: raise ProtocolError('Expected one chat choice')
        c=choices[0]; message=c.get('message')
        if not isinstance(message,dict) or message.get('role')!='assistant': raise ProtocolError('Missing assistant message')
        calls=[]; seen=set()
        raw_calls=message.get('tool_calls') or []
        if not isinstance(raw_calls,list): raise ProtocolError('tool_calls must be list')
        for index,tc in enumerate(raw_calls):
            if not isinstance(tc,dict) or tc.get('type')!='function' or not isinstance(tc.get('function'),dict):
                raise ProtocolError('Invalid tool call')
            native_id=tc.get('id'); fn=tc['function']
            if not isinstance(native_id,str) or not native_id or native_id in seen: raise ProtocolError('Missing/duplicate tool_call_id')
            seen.add(native_id)
            if not isinstance(fn.get('name'),str) or not isinstance(fn.get('arguments'),str): raise ProtocolError('Invalid function')
            args=parse_json(fn['arguments'].encode('utf-8'))
            calls.append(Call(f'{request_id}:c{index}',fn['name'],args,native_id))
        finish=c.get('finish_reason')
        if finish not in ('stop','tool_calls'): raise ProtocolError('Non-complete chat choice: '+str(finish))
        if bool(calls)!=(finish=='tool_calls'): raise ProtocolError('tool_calls/finish_reason mismatch')
        return Reply(tuple(calls),copy.deepcopy(message),value.get('usage'),value.get('system_fingerprint') or value.get('model'),finish,message.get('content'))
    def accept(self,conversation,reply,results):
        new=copy.deepcopy(conversation)
        new['messages'].append(copy.deepcopy(reply.native_message))
        for call,result in results:
            new['messages'].append({'role':'tool','tool_call_id':call.native_id,
                'content':json.dumps(result.observation,ensure_ascii=False,allow_nan=False)})
            if result.image:
                new['messages'].append({'role':'user','content':[
                    {'type':'text','text':f'Image returned by tool call {call.native_id}'},
                    {'type':'image_url','image_url':{'url':f'data:{result.image_mime};base64,'+base64.b64encode(result.image).decode('ascii')}}]})
        return new
