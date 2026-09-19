# Errors, retries and rate limits

## Authentication

Access tokens are obtained with the client credentials grant using a client id
and secret, and expire after a fixed period. A `401` means the token is missing,
expired or invalid: refresh it once and retry. A `403` means the credentials are
valid but lack permission for the operation, which retrying will not fix.

## Idempotency

Write operations accept a request id header so that a retried request is not
processed twice. Reuse the same request id when retrying a failed write, and use
a fresh one for a genuinely new operation. Keys are typically retained for
several hours.

## HTTP status meanings

- `400` / `422` - the request is malformed or violates a business rule. The
  response details name the offending field or issue code.
- `404` - the resource does not exist, or the identifier belongs to another
  account or environment.
- `409` - a conflict with the current state of the resource.
- `429` - rate limited. Respect the `Retry-After` header when present.
- `5xx` - a transient server problem. Retry with exponential backoff.

## Reading error payloads

Errors carry a `name` (the error class), a human readable `message`, and a
`details` array. Each detail has an `issue` code and often a `field` pointer.
The issue code is the machine-readable reason and is the right thing to branch
on.

Issue codes that mention a missing precondition, such as a resource not being
captured or an order not being approved, indicate the workflow ran in the wrong
order. The fix is to perform the prerequisite operation, not to retry.

Issue codes that mention an action already having happened indicate a business
outcome. Report it rather than retrying.

## Sandbox versus live

Sandbox and live use different base URLs and different credentials. Identifiers
are not portable between them; a sandbox id used against live returns `404`.
