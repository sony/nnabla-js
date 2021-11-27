export function getOrThrow<T>(data: T | undefined | null): T {
  if (data === undefined || data === null) {
    throw Error('got undefined variable.');
  } else {
    return data as T;
  }
}

export function getAsArrayOrThrow<T>(data: T[] | undefined | null): T[] {
  if (data === undefined || data === null) {
    throw Error('got undefined variable.');
  } else {
    if (Array.isArray(data)) {
      return data as T[];
    }
    return [data] as T[];
  }
}
