/* eslint-disable no-param-reassign */

const tokenList = ['fieldName', 'string', 'number', 'openBrace', 'closeBrace', 'colon', 'eof'];
type Token = typeof tokenList[number];

function checkWhiteSpace(chr: string): boolean {
  return chr === ' ' || chr === '\n';
}

function snakeToCamel(name: string): string {
  const camel = name
    .split('_')
    .map((s) => s.charAt(0).toUpperCase() + s.substring(1))
    .join('');
  return camel.charAt(0).toLowerCase() + camel.substring(1);
}

class Tokenizer {
  text: string;

  cursor: number;

  lineNo: number;

  constructor(text: string) {
    this.text = text;
    this.cursor = 0;
    this.lineNo = 1;
  }

  readString(): string {
    if (this.text[this.cursor] !== '"') {
      throw Error('invalid symbol');
    }
    this.cursor += 1;

    const startCursor = this.cursor;
    while (this.text[this.cursor] !== '"') {
      this.cursor += 1;
    }

    const str = this.text.substring(startCursor, this.cursor);
    this.cursor += 1;

    return str;
  }

  readNumberOrFieldName(): string {
    const startCursor = this.cursor;
    while (!checkWhiteSpace(this.text[this.cursor]) && this.text[this.cursor] !== ':') {
      this.cursor += 1;
    }
    return this.text.substring(startCursor, this.cursor);
  }

  advance(): [Token, string] {
    while (!this.finished()) {
      const chr = this.text[this.cursor];
      let value = '';

      switch (chr) {
        case '\n':
          this.lineNo += 1;
          this.cursor += 1;
          break;
        case ' ':
          this.cursor += 1;
          break;
        case '{':
          this.cursor += 1;
          return ['openBrace', '{'];
        case '}':
          this.cursor += 1;
          return ['closeBrace', '}'];
        case ':':
          this.cursor += 1;
          return ['colon', ':'];
        case '"':
          return ['string', this.readString()];
        default:
          // number of field name
          value = this.readNumberOrFieldName();
          if (/^-?(\d|\.)+$/.test(value)) {
            return ['number', value];
          }
          return ['fieldName', value];
      }
    }

    return ['eof', ''];
  }

  finished(): boolean {
    return this.cursor >= this.text.length;
  }
}

function parse(tokenizer: Tokenizer, obj: { [key: string]: any }): { [key: string]: any } {
  while (!tokenizer.finished()) {
    const [token, value] = tokenizer.advance();

    if (token === 'eof' || token === 'closeBrace') {
      return obj;
    }

    if (token === 'fieldName') {
      const [nextToken, nextValue] = tokenizer.advance();

      if (nextToken === 'openBrace') {
        const fieldName = snakeToCamel(value);
        const childObj = {};
        parse(tokenizer, childObj);
        if (Object.prototype.hasOwnProperty.call(obj, fieldName)) {
          if (Array.isArray(obj[fieldName])) {
            obj[fieldName].push(childObj);
          } else {
            obj[fieldName] = [obj[fieldName], childObj];
          }
        } else {
          obj[fieldName] = childObj;
        }
      } else if (nextToken === 'colon') {
        const fieldName = snakeToCamel(value);
        const [valueType, rawValue] = tokenizer.advance();
        let fieldValue: string | number = '';
        if (valueType === 'string') {
          fieldValue = rawValue;
        } else if (valueType === 'number') {
          fieldValue = Number(rawValue);
        } else {
          throw Error(`invalid token: ${rawValue} (${valueType}) at line ${tokenizer.lineNo}`);
        }
        if (Object.prototype.hasOwnProperty.call(obj, fieldName)) {
          if (Array.isArray(obj[fieldName])) {
            obj[fieldName].push(fieldValue);
          } else {
            obj[fieldName] = [obj[fieldName], fieldValue];
          }
        } else {
          obj[fieldName] = fieldValue;
        }
      } else {
        throw Error(`invalid token: ${nextValue} (${nextToken}) at line ${tokenizer.lineNo}`);
      }
    }
  }

  return obj;
}

export default function decodePbtxt(data: string): { [key: string]: any } {
  const tokenizer = new Tokenizer(data);
  return parse(tokenizer, {});
}
