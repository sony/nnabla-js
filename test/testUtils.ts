export function expectClose(x: number, y: number, atol: number): void {
  expect(x - y).toBeLessThan(atol);
  expect(x - y).toBeGreaterThan(-atol);
}

export function expectAllClose(x: number[], y: number[], atol: number): void {
  expect(x.length).toBe(y.length);
  for (let i = 0; i < x.length; i += 1) {
    expectClose(x[i], y[i], atol);
  }
}
