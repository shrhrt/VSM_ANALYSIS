export interface Suggestion {
  insert:  string;   // 挿入するテキスト
  label:   string;   // ドロップダウンに表示する名称
  aliases: string[]; // 検索キーワード（小文字）
}

export const SUGGESTIONS: Suggestion[] = [
  // ── 酸化鉄 ──────────────────────────────────────
  { insert: "Fe₃O₄",      label: "マグネタイト",                 aliases: ["fe3o4", "magnetite", "magnet", "fe"] },
  { insert: "γ-Fe₂O₃",   label: "マグヘマイト",                 aliases: ["fe2o3", "maghemite", "maghe", "gamma", "gfe", "fe"] },
  { insert: "α-Fe₂O₃",   label: "ヘマタイト",                   aliases: ["fe2o3", "hematite", "hema", "alpha", "afe", "fe"] },
  { insert: "ε-Fe₂O₃",   label: "ε-酸化鉄",                    aliases: ["fe2o3", "epsilon", "efe", "fe"] },
  { insert: "FeO",         label: "酸化鉄 (II)",                  aliases: ["feo", "fe"] },

  // ── スピネルフェライト ───────────────────────────
  { insert: "CoFe₂O₄",   label: "コバルトフェライト",            aliases: ["cofe", "cobalt ferrite", "cfo", "co"] },
  { insert: "NiFe₂O₄",   label: "ニッケルフェライト",            aliases: ["nife", "nickel ferrite", "nfo", "ni"] },
  { insert: "MnFe₂O₄",   label: "マンガンフェライト",            aliases: ["mnfe", "manganese ferrite", "mfo", "mn"] },
  { insert: "MgFe₂O₄",   label: "マグネシウムフェライト",        aliases: ["mgfe", "magnesium ferrite", "mgfo", "mg"] },
  { insert: "ZnFe₂O₄",   label: "亜鉛フェライト",               aliases: ["znfe", "zinc ferrite", "zfo", "zn"] },
  { insert: "CuFe₂O₄",   label: "銅フェライト",                 aliases: ["cufe", "copper ferrite", "cu"] },
  { insert: "Li₀.₅Fe₂.₅O₄", label: "リチウムフェライト",        aliases: ["life", "lithium ferrite", "li"] },

  // ── 六方晶フェライト ─────────────────────────────
  { insert: "BaFe₁₂O₁₉", label: "バリウムフェライト (BaM)",    aliases: ["bafe", "barium ferrite", "bam", "ba"] },
  { insert: "SrFe₁₂O₁₉", label: "ストロンチウムフェライト (SrM)", aliases: ["srfe", "strontium ferrite", "srm", "sr"] },

  // ── 単体金属 ─────────────────────────────────────
  { insert: "Fe",  label: "鉄 (iron)",          aliases: ["iron", "fe"] },
  { insert: "Co",  label: "コバルト (cobalt)",   aliases: ["cobalt", "co"] },
  { insert: "Ni",  label: "ニッケル (nickel)",   aliases: ["nickel", "ni"] },
  { insert: "Mn",  label: "マンガン (manganese)", aliases: ["manganese", "mn"] },
  { insert: "Gd",  label: "ガドリニウム",        aliases: ["gadolinium", "gd"] },
  { insert: "Dy",  label: "ジスプロシウム",      aliases: ["dysprosium", "dy"] },
  { insert: "Nd",  label: "ネオジム",            aliases: ["neodymium", "nd"] },
  { insert: "Sm",  label: "サマリウム",          aliases: ["samarium", "sm"] },
  { insert: "Tb",  label: "テルビウム",          aliases: ["terbium", "tb"] },

  // ── 非磁性金属（積層用） ──────────────────────────
  { insert: "Pt",  label: "白金 (platinum)",    aliases: ["platinum", "pt"] },
  { insert: "Pd",  label: "パラジウム",         aliases: ["palladium", "pd"] },
  { insert: "Cr",  label: "クロム (chromium)",  aliases: ["chromium", "cr"] },
  { insert: "Cu",  label: "銅 (copper)",        aliases: ["copper", "cu"] },
  { insert: "Ta",  label: "タンタル",           aliases: ["tantalum", "ta"] },
  { insert: "Ru",  label: "ルテニウム",         aliases: ["ruthenium", "ru"] },
  { insert: "Ti",  label: "チタン",             aliases: ["titanium", "ti"] },
  { insert: "W",   label: "タングステン",        aliases: ["tungsten", "w"] },
  { insert: "Au",  label: "金 (gold)",          aliases: ["gold", "au"] },
  { insert: "Ag",  label: "銀 (silver)",        aliases: ["silver", "ag"] },
  { insert: "Al",  label: "アルミニウム",        aliases: ["aluminum", "al"] },
  { insert: "Mo",  label: "モリブデン",         aliases: ["molybdenum", "mo"] },
  { insert: "Hf",  label: "ハフニウム",         aliases: ["hafnium", "hf"] },

  // ── 基板・絶縁体 ─────────────────────────────────
  { insert: "MgO",     label: "酸化マグネシウム",             aliases: ["mgo", "magnesium oxide", "mg"] },
  { insert: "Al₂O₃",  label: "アルミナ (sapphire)",          aliases: ["al2o3", "alumina", "sapphire", "al"] },
  { insert: "SiO₂",   label: "二酸化ケイ素 (silica)",         aliases: ["sio2", "silica", "si"] },
  { insert: "Si",      label: "シリコン",                     aliases: ["silicon", "si"] },
  { insert: "SrTiO₃", label: "チタン酸ストロンチウム (STO)",  aliases: ["srtio3", "sto", "sr"] },
  { insert: "GaAs",   label: "ガリウムヒ素",                 aliases: ["gaas", "gallium arsenide", "ga"] },
  { insert: "GaN",    label: "窒化ガリウム",                 aliases: ["gan", "gallium nitride", "ga"] },
  { insert: "MgAl₂O₄", label: "スピネル基板",               aliases: ["mgal2o4", "spinel", "mg"] },
  { insert: "LaAlO₃", label: "LAO基板",                     aliases: ["lao", "laalo3", "lanthanum", "la"] },
  { insert: "TiN",    label: "窒化チタン",                   aliases: ["tin", "titanium nitride", "ti"] },

  // ── 合金 ─────────────────────────────────────────
  { insert: "FePt",      label: "FePt合金",                  aliases: ["fept", "fe", "pt"] },
  { insert: "CoPt",      label: "CoPt合金",                  aliases: ["copt", "co", "pt"] },
  { insert: "FePd",      label: "FePd合金",                  aliases: ["fepd", "fe"] },
  { insert: "FeRh",      label: "FeRh合金",                  aliases: ["ferh", "fe"] },
  { insert: "NdFeB",     label: "ネオジム磁石",              aliases: ["ndfeb", "nd", "neodymium"] },
  { insert: "SmCo",      label: "サマリウムコバルト磁石",    aliases: ["smco", "sm", "co"] },
  { insert: "Permalloy", label: "パーマロイ (Ni₈₀Fe₂₀)",    aliases: ["permalloy", "py", "nife", "ni"] },
  { insert: "Py",        label: "パーマロイ略称 (Py)",       aliases: ["py", "permalloy"] },
  { insert: "Heusler",   label: "ホイスラー合金",            aliases: ["heusler"] },
  { insert: "Co₂MnSi",  label: "コバルトマンガンシリサイド", aliases: ["co2mnsi", "cms", "co"] },
  { insert: "Co₂FeAl",  label: "コバルト鉄アルミ",          aliases: ["co2feal", "cfa", "co"] },

  // ── ギリシャ文字 ─────────────────────────────────
  { insert: "α",  label: "アルファ (alpha)",        aliases: ["alpha", "alph"] },
  { insert: "β",  label: "ベータ (beta)",            aliases: ["beta"] },
  { insert: "γ",  label: "ガンマ (gamma)",           aliases: ["gamma", "gamm"] },
  { insert: "δ",  label: "デルタ (delta)",           aliases: ["delta", "delt"] },
  { insert: "ε",  label: "イプシロン (epsilon)",     aliases: ["epsilon", "eps"] },
  { insert: "μ",  label: "ミュー (mu)",              aliases: ["mu"] },
  { insert: "χ",  label: "カイ / 磁化率 (chi)",      aliases: ["chi", "susceptibility", "susc"] },
  { insert: "λ",  label: "ラムダ (lambda)",          aliases: ["lambda", "lamb"] },
  { insert: "θ",  label: "シータ (theta)",           aliases: ["theta", "thet"] },
  { insert: "σ",  label: "シグマ (sigma)",           aliases: ["sigma", "sigm"] },
  { insert: "Δ",  label: "デルタ大文字 (Delta)",     aliases: ["delta", "delt"] },
  { insert: "Ω",  label: "オメガ大文字 (Omega)",     aliases: ["omega", "omeg"] },

  // ── 用語 ─────────────────────────────────────────
  { insert: "sub.",  label: "基板 (substrate)",  aliases: ["sub", "substrate"] },
  { insert: "buf.",  label: "バッファ層 (buffer)", aliases: ["buf", "buffer"] },
  { insert: "seed",  label: "シード層",           aliases: ["seed"] },
  { insert: "cap",   label: "キャップ層",         aliases: ["cap"] },
  { insert: "ann.",  label: "アニール (anneal)",  aliases: ["ann", "anneal"] },
  { insert: "RT",    label: "室温 (room temp.)",  aliases: ["rt", "room", "temp"] },
];

/** 現在のトークンに対してスコア順でマッチする候補を返す */
export function searchSuggestions(token: string, max = 7): Suggestion[] {
  if (token.length < 2) return [];
  const q = token.toLowerCase();

  const scored = SUGGESTIONS.flatMap((s) => {
    const insertLow = s.insert.toLowerCase();
    if (insertLow.startsWith(q))                       return [{ s, score: 4 }];
    if (s.aliases.some((a) => a.startsWith(q)))        return [{ s, score: 3 }];
    if (insertLow.includes(q))                         return [{ s, score: 2 }];
    if (s.aliases.some((a) => a.includes(q)))          return [{ s, score: 1 }];
    if (s.label.includes(token))                       return [{ s, score: 1 }];
    return [];
  });

  return scored
    .sort((a, b) => b.score - a.score)
    .map((x) => x.s)
    .slice(0, max);
}

/** 入力値とカーソル位置から「現在のトークン」と開始位置を返す */
export function getCurrentToken(value: string, cursor: number): { token: string; start: number } {
  const before = value.slice(0, cursor);
  // /  (  スペース を区切り文字とする
  const lastSep = Math.max(
    before.lastIndexOf("/"),
    before.lastIndexOf(" "),
    before.lastIndexOf("("),
    before.lastIndexOf(","),
    -1,
  );
  const start = lastSep + 1;
  return { token: before.slice(start), start };
}
