# Payment lifecycle: authorize, capture, refund

A payment moves through distinct states, and most integration errors come from
acting on the wrong state.

## Order

An order represents the buyer's intent to pay. Creating an order returns an
`order_id`. An order carries an `intent` of either `CAPTURE` or `AUTHORIZE`.

- `intent=CAPTURE` means money moves as soon as the order is captured.
- `intent=AUTHORIZE` means funds are held first and captured later.

An order must be approved by the buyer before it can be authorized or captured.
Acting on an unapproved order fails with `ORDER_NOT_APPROVED`.

## Authorization

Authorizing an order places a hold on the buyer's funds and returns an
`authorization_id`. Authorizations expire; a typical honour period is three
days, with a maximum of roughly 29 days. An expired authorization must be
reauthorized before it can be captured.

Voiding an authorization releases the hold. A voided authorization cannot be
captured.

## Capture

Capturing an authorization moves the money and returns a `capture_id`. A
capture may be partial; multiple captures against one authorization are
possible when the authorization is marked as reusable.

Refunds operate on captures, never on authorizations or orders. Attempting to
refund a payment that has not been captured fails, commonly with an issue code
containing `NOT_CAPTURED`.

## Refund

Refunding a capture returns a `refund_id`. Refunds can be full or partial, and
the total refunded cannot exceed the captured amount. A capture that has been
fully refunded rejects further refunds with `CAPTURE_FULLY_REFUNDED`.

## Typical sequences

- Immediate sale: create order (`intent=CAPTURE`) -> buyer approves -> capture order -> refund capture
- Delayed settlement: create order (`intent=AUTHORIZE`) -> buyer approves -> authorize order -> capture authorization -> refund capture

## Identifier dependencies

- capturing an order requires `order_id` and produces `capture_id`
- capturing an authorization requires `authorization_id` and produces `capture_id`
- refunding requires `capture_id` and produces `refund_id`
- refund lookups require `refund_id`
