/**
 * Chinese display labels for the backend enum values.
 *
 * The API stores and returns raw English enum values (see api/models/user.py
 * and api/models/clothing.py). Rendering those directly in the UI is what made
 * the profile page show "male" / "average" / "medium". Every UI surface should
 * go through the helpers here instead.
 */

// api/models/user.py :: GenderEnum
export const GENDER_LABELS: Record<string, string> = {
  male: '男',
  female: '女',
  other: '其他',
}

// api/models/user.py :: BodyShapeEnum
export const BODY_SHAPE_LABELS: Record<string, string> = {
  slim: '纤细型',
  athletic: '运动型',
  average: '标准型',
  curvy: '曲线型',
  plus_size: '大码型',
}

export const BODY_SHAPE_COLORS: Record<string, string> = {
  slim: 'cyan',
  athletic: 'blue',
  average: 'green',
  curvy: 'orange',
  plus_size: 'red',
}

// api/models/user.py :: SkinToneEnum — this is skin *tone* (肤色), not skin type (肤质).
export const SKIN_TONE_LABELS: Record<string, string> = {
  fair: '白皙',
  light: '偏白',
  medium: '自然色',
  tan: '小麦色',
  dark: '深色',
}

// Approximate swatch colours for the skin tone scale.
export const SKIN_TONE_SWATCHES: Record<string, string> = {
  fair: '#f7e0d0',
  light: '#f0cdb4',
  medium: '#d9a67e',
  tan: '#b57a4e',
  dark: '#7a4a2b',
}

// api/models/clothing.py :: StyleEnum
export const STYLE_LABELS: Record<string, string> = {
  minimalist: '简约',
  casual: '休闲',
  bohemian: '波西米亚',
  vintage: '复古',
  streetwear: '街头',
  business: '商务',
  sporty: '运动',
  romantic: '甜美',
}

// api/models/clothing.py :: CategoryEnum
export const CATEGORY_LABELS: Record<string, string> = {
  top: '上衣',
  bottom: '下装',
  dress: '连衣裙',
  outerwear: '外套',
  shoes: '鞋履',
  accessories: '配饰',
}

export const CATEGORY_EMOJIS: Record<string, string> = {
  top: '👕',
  bottom: '👖',
  dress: '👗',
  outerwear: '🧥',
  shoes: '👟',
  accessories: '👜',
}

// Interaction relationship types returned by /interactions/history/{user_id}
export const RELATION_LABELS: Record<string, string> = {
  VIEWED: '浏览',
  BOUGHT: '购买',
  RATED: '评分',
  PREFERS: '偏好',
  HAS_STYLE: '风格关联',
}

export const RELATION_COLORS: Record<string, string> = {
  VIEWED: 'default',
  BOUGHT: 'red',
  RATED: 'gold',
  PREFERS: 'blue',
  HAS_STYLE: 'purple',
}

const COLOR_HEX: Record<string, string> = {
  black: '#000000',
  white: '#FFFFFF',
  blue: '#1890ff',
  navy: '#001f5c',
  grey: '#8c8c8c',
  gray: '#8c8c8c',
  red: '#f5222d',
  green: '#52c41a',
  yellow: '#faad14',
  purple: '#722ed1',
  pink: '#eb2f96',
  orange: '#fa8c16',
  brown: '#8b4513',
  beige: '#f5f5dc',
}

const COLOR_LABELS: Record<string, string> = {
  black: '黑色',
  white: '白色',
  blue: '蓝色',
  navy: '藏青',
  grey: '灰色',
  gray: '灰色',
  red: '红色',
  green: '绿色',
  yellow: '黄色',
  purple: '紫色',
  pink: '粉色',
  orange: '橙色',
  brown: '棕色',
  beige: '米色',
}

// Chinese colour names also appear in the graph, so map both directions.
const CN_COLOR_HEX: Record<string, string> = Object.entries(COLOR_LABELS).reduce(
  (acc, [en, cn]) => {
    acc[cn] = COLOR_HEX[en]
    return acc
  },
  {} as Record<string, string>
)

/** Look up a label table, falling back to the raw value rather than blanking it. */
export function labelOf(
  table: Record<string, string>,
  value?: string | null,
  fallback = '-'
): string {
  if (value === undefined || value === null || value === '') return fallback
  return table[value] ?? table[String(value).toLowerCase()] ?? String(value)
}

export const genderLabel = (v?: string | null) => labelOf(GENDER_LABELS, v)
export const bodyShapeLabel = (v?: string | null) => labelOf(BODY_SHAPE_LABELS, v, '未设置')
export const skinToneLabel = (v?: string | null) => labelOf(SKIN_TONE_LABELS, v)
export const styleLabel = (v?: string | null) => labelOf(STYLE_LABELS, v)
export const categoryLabel = (v?: string | null) => labelOf(CATEGORY_LABELS, v)
export const relationLabel = (v?: string | null) => labelOf(RELATION_LABELS, v)

export function colorLabel(value?: string | null): string {
  if (!value) return '未知'
  const key = value.toLowerCase()
  return COLOR_LABELS[key] ?? value
}

export function colorHex(value?: string | null): string {
  if (!value) return '#d9d9d9'
  const key = value.toLowerCase()
  return COLOR_HEX[key] ?? CN_COLOR_HEX[value] ?? '#d9d9d9'
}
