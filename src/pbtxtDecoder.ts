/* eslint-disable no-param-reassign */
import * as nnp from './proto/nnabla_pb';

const tokenList = [
  'fieldName',
  'string',
  'number',
  'openBrace',
  'closeBrace',
  'colon',
  'eof',
  'boolean',
];
type Token = typeof tokenList[number];

function checkWhiteSpace(chr: string): boolean {
  return chr === ' ' || chr === '\n';
}

function snakeToCamel(name: string): string {
  return name
    .split('_')
    .map((s) => s.charAt(0).toUpperCase() + s.substring(1))
    .join('');
}

function convertParamField(name: string): string {
  return `${name}eter`;
}

function convertSpecialField(name: string): string {
  if (name.includes('Relu')) {
    return name.replace('Relu', 'ReLU');
  }
  if (name === 'Pad' || name === 'Stride' || name === 'Dilation' || name === 'Kernel') {
    return 'Shape';
  }
  return name;
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

  readDataOrFieldName(): string {
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
          value = this.readDataOrFieldName();
          if (/^-?(\d|\.)+$/.test(value)) {
            return ['number', value];
          }
          if (value === 'true' || value === 'false') {
            return ['boolean', value];
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

function parse(tokenizer: Tokenizer, obj: any): void {
  const proto = Object.getPrototypeOf(obj);

  while (!tokenizer.finished()) {
    const [token, value] = tokenizer.advance();

    if (token === 'eof' || token === 'closeBrace') {
      return;
    }

    if (token === 'fieldName') {
      const [nextToken, nextValue] = tokenizer.advance();

      if (nextToken === 'openBrace') {
        const fieldName = snakeToCamel(value);
        let childObj: any = {};
        if (Object.prototype.hasOwnProperty.call(proto, `add${fieldName}`)) {
          childObj = proto[`add${fieldName}`].call(obj);
        } else {
          childObj = proto[`get${fieldName}`].call(obj);
        }
        if (childObj === undefined) {
          let typeName = fieldName;
          if (typeName.endsWith('Param')) {
            typeName = convertParamField(typeName);
          }
          typeName = convertSpecialField(typeName);
          childObj = new (nnp as any)[typeName]();
          proto[`set${fieldName}`].call(obj, childObj);
        }
        parse(tokenizer, childObj);
      } else if (nextToken === 'colon') {
        const fieldName = snakeToCamel(value);
        const [valueType, rawValue] = tokenizer.advance();
        let fieldValue: string | number | boolean = '';
        if (valueType === 'string') {
          fieldValue = rawValue;
        } else if (valueType === 'number') {
          fieldValue = Number(rawValue);
        } else if (valueType === 'boolean') {
          fieldValue = Boolean(rawValue);
        } else {
          throw Error(`invalid token: ${rawValue} (${valueType}) at line ${tokenizer.lineNo}`);
        }
        if (Object.prototype.hasOwnProperty.call(proto, `add${fieldName}`)) {
          proto[`add${fieldName}`].call(obj, fieldValue);
        } else {
          proto[`set${fieldName}`].call(obj, fieldValue);
        }
      } else {
        throw Error(`invalid token: ${nextValue} (${nextToken}) at line ${tokenizer.lineNo}`);
      }
    }
  }
}

export default function decodePbtxt(data: string, obj: any): void {
  const tokenizer = new Tokenizer(data);
  parse(tokenizer, obj);
}
