"""Cash-secured standard option writing in explicitly approved Roth profiles.

All cash and underlying shares live in PortfolioLedger. Premium is an asset
offset by the outstanding short liability, never free excess return. Assignment
and nonassignment require supplied notices; no synthetic quote or automatic
optimal-exercise/assignment rule is inferred. Taxable/adjusted/spread cases
remain gated independently of these mechanics.
"""
from datetime import date
from zoneinfo import ZoneInfo

from letf.accounts import calendar_date,positive
from letf.accuracy import UnsupportedModelError
from letf.derivatives import LongOptionBook


class SecuredOptionBook(LongOptionBook):
    def __init__(self,ledger,account_id,*,approved_strategies,permission_source,permission_verified_on,
                 commission_per_contract=0.):
        if not permission_source or permission_verified_on is None:
            raise UnsupportedModelError('Writing requires separately documented strategy-level broker permissions')
        if not approved_strategies or set(approved_strategies)-{'covered_call','cash_secured_put'}:
            raise UnsupportedModelError('Only explicitly approved covered-call/cash-secured-put subset')
        if any(isinstance(b,SecuredOptionBook) and b.account_id==account_id for b in ledger.derivative_books):
            raise ValueError('Use one aggregate secured-writing book per account')
        self.approved=frozenset(approved_strategies);self.permission_source=permission_source
        self.permission_date=calendar_date(permission_verified_on)
        super().__init__(ledger,account_id,commission_per_contract=commission_per_contract)

    def _key(self,symbol):return 'secured_options:'+symbol

    def _reserves(self,contract,quantity):
        account=self.ledger.accounts[self.account_id];key=self._key(contract.symbol)
        account.collateral_reservations.pop(key,None);account.share_reservations.pop(key,None)
        if quantity:
            if contract.right=='put':account.collateral_reservations[key]=contract.strike*contract.multiplier*quantity
            else:account.share_reservations[key]={contract.underlying:contract.multiplier*quantity}

    def _standard(self,contract,when):
        strategy='covered_call' if contract.right=='call' else 'cash_secured_put'
        if strategy not in self.approved or when<self.permission_date:
            raise UnsupportedModelError('Writing strategy lacks effective broker approval')
        if when<date(2024,5,28):raise UnsupportedModelError('This physical-delivery writing subset uses verified modern T+1 only')
        if contract.style!='american' or contract.settlement!='physical' or contract.multiplier!=100 or contract.adjusted:
            raise UnsupportedModelError('Only standard100-share American physical contracts are implemented here')
        if contract.underlying not in self.ledger.securities:
            raise UnsupportedModelError('Underlying security requires its own reviewed definition')
        if when>contract.expiry:raise ValueError('Option expired')
        for book in self.ledger.derivative_books:
            if book is not self and contract.symbol in getattr(book,'positions',{}):
                raise UnsupportedModelError('Offsetting same-contract books require one reviewed netting/spread handler')

    def trade(self,timestamp,contract,side,contracts,quote):
        if quote.contract_symbol!=contract.symbol:raise ValueError('Quote does not match dated contract')
        price=quote.fill(timestamp,side,contracts)
        if self.last_timestamp is not None and timestamp<self.last_timestamp:
            raise ValueError('Option executions must be chronological within each session')
        when=timestamp.astimezone(ZoneInfo('America/New_York')).date();self._standard(contract,when)
        old_contract,old=self.positions.get(contract.symbol,(contract,0))
        if old_contract!=contract:raise ValueError('Contract specification changed; OCC action required')
        if old:self._check_actions(contract.symbol)
        if side=='BUY' and contracts>old:raise ValueError('Buy-to-close cannot exceed short contracts')
        settle=self.settlement_date(when);self.ledger._advance(when)
        account=self.ledger.accounts[self.account_id];fee=contracts*self.commission
        new=old+contracts if side=='SELL' else old-contracts
        own=account.collateral_reservations.get(self._key(contract.symbol),0.)
        others=sum(account.collateral_reservations.values())-own
        reserve=contract.strike*100*new if contract.right=='put' else 0.
        if side=='SELL':
            if account.cash-others-reserve<fee-1e-9:raise ValueError('Insufficient settled cash for full strike collateral/commission')
            if contract.right=='call':
                encumbered=sum(v.get(contract.underlying,0.) for v in account.share_reservations.values())
                pending_buys=sum(e['shares'] for e in self.ledger.events if e.get('account')==self.account_id and
                    e.get('security')==contract.underlying and e['action']=='BUY' and e.get('settlement_date',when)>when)
                if self.ledger.quantity(self.account_id,contract.underlying)-encumbered-pending_buys<contracts*100-1e-9:
                    raise ValueError('Insufficient settled unencumbered shares for covered call')
            account.cash-=fee
            account.unsettled.append((settle,price*contracts*100))
        else:
            debit=price*contracts*100+fee
            if account.cash-others-reserve<debit-1e-9:raise ValueError('Insufficient cash for short close while preserving other collateral')
            account.cash-=debit
        self._reserves(contract,new)
        if new:
            self.positions[contract.symbol]=(contract,new)
            if not old:self.action_cursor[contract.symbol]=len(self.ledger.events)
        else:self.positions.pop(contract.symbol,None);self.action_cursor.pop(contract.symbol,None)
        self.last_timestamp=timestamp
        event={'date':when,'timestamp':timestamp,'account':self.account_id,'action':'SECURED_OPTION_FILL',
            'contract':contract.symbol,'side':side,'contracts':contracts,'price':price,'commission':fee,
            'settlement_date':settle,'source':quote.source,'permission_source':self.permission_source,
            'execution_exact':False,'classification':'CONDITIONAL_STRESS'}
        self.ledger.events.append(event);return event

    def mark_value(self,prices):
        value=0.
        for symbol,(contract,quantity) in self.positions.items():
            self._check_actions(symbol);positive(prices[symbol],'short-option liquidation ask',True)
            value-=prices[symbol]*quantity*100
        return value

    def mark(self,timestamp,quotes):
        if timestamp.tzinfo is None:raise ValueError('Timezone-aware mark required')
        if self.last_timestamp is not None and timestamp<self.last_timestamp:raise ValueError('Mark precedes latest trade')
        value=0.
        for symbol,(contract,quantity) in self.positions.items():
            self._check_actions(symbol);quote=quotes[symbol]
            if quote.contract_symbol!=symbol:raise ValueError('Wrong liability quote')
            self._standard(contract,timestamp.astimezone(ZoneInfo('America/New_York')).date())
            quote.fill(timestamp,'BUY',1);value-=quote.ask*quantity*100
        return value

    def exercise(self,*args,**kwargs):
        raise UnsupportedModelError('A writer cannot exercise; use an actual assignment notice')

    def assign(self,when,symbol,contracts,*,notice_source,assignment_fee=0.):
        when=calendar_date(when);positive(assignment_fee,'assignment fee',True)
        contract,old=self.positions[symbol];self._check_actions(symbol);self._standard(contract,when)
        if not notice_source:raise ValueError('Explicit broker/OCC assignment notice required')
        if type(contracts)is not int or not 0<contracts<=old:raise ValueError('Invalid assignment contract count')
        self._reserves(contract,old-contracts)
        try:
            event=self.ledger.execute(when,self.account_id,contract.underlying,
                'SELL' if contract.right=='call' else 'BUY',contracts*100,contract.strike,
                settlement_date=self.settlement_date(when),execution_kind='assignment',
                execution_source=notice_source,commission_override=assignment_fee)
        except Exception:
            self._reserves(contract,old);raise
        if old>contracts:self.positions[symbol]=(contract,old-contracts)
        else:self.positions.pop(symbol);self.action_cursor.pop(symbol,None)
        self.ledger.events.append({'date':when,'account':self.account_id,'action':'OPTION_ASSIGNMENT',
            'contract':symbol,'contracts':contracts,'source':notice_source,'settlement_date':event['settlement_date'],
            'automatic_assignment_assumed':False})
        return self.ledger.events[-1]

    def expire(self,when,symbol,settlement_price=None,*,nonassignment_source=None,**kwargs):
        when=calendar_date(when);contract,quantity=self.positions[symbol]
        self._check_actions(symbol)
        if when!=contract.expiry or not nonassignment_source:
            raise UnsupportedModelError('Expiry requires explicit final nonassignment evidence; moneyness alone is insufficient')
        self.ledger._advance(when);self._reserves(contract,0)
        self.positions.pop(symbol);self.action_cursor.pop(symbol,None)
        self.ledger.events.append({'date':when,'account':self.account_id,'action':'SHORT_OPTION_EXPIRY',
            'contract':symbol,'contracts':quantity,'source':nonassignment_source,'assignment_inferred_from_price':False})
